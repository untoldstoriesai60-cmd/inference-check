"""Run synthetic Chat Completions compatibility probes outside the app's request path."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from http.client import HTTPException
import json
import os
from pathlib import Path
import sys
import tempfile
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

VERSION = "0.1.0"
SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
    "required": ["city", "days"],
    "additionalProperties": False,
}
EXPECTED = {"city": "Recife", "days": 2}
FUNCTION = {
    "name": "get_forecast",
    "description": "Get the forecast for a city and number of days.",
    "parameters": SCHEMA,
    "strict": True,
}
CHECKS = ("structured_output", "tool_call", "streaming_tool_call")
MAX_BYTES = 1_048_576


class ProbeFailure(Exception):
    """A diagnostic that can be recorded without copying upstream content."""


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request_payload(route, check):
    structured = check == "structured_output"
    payload = {
        "model": route["model"],
        "messages": [{
            "role": "user",
            "content": (
                'Return city "Recife" and days 2 in the requested JSON schema.'
                if structured else 'Call get_forecast for "Recife" with days set to 2.'
            ),
        }],
        "stream": check == "streaming_tool_call",
    }
    if structured:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "forecast_request", "schema": SCHEMA, "strict": True},
        }
    else:
        payload["tools"] = [{"type": "function", "function": FUNCTION}]
        payload["tool_choice"] = {"type": "function", "function": {"name": FUNCTION["name"]}}
    if route["upstream"]:
        payload["provider"] = {
            "only": [route["upstream"]],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
    return payload


def check_fields(value):
    # This probe uses a fixed schema with two required fields and no extras.
    if (
        not isinstance(value, dict)
        or value.keys() != EXPECTED.keys()
        or not isinstance(value["city"], str)
        or type(value["days"]) is not int
    ):
        raise ProbeFailure("schema_mismatch")
    if value != EXPECTED:
        raise ProbeFailure("unexpected_values")


def check_tool(call):
    function = call["function"]
    if call["type"] != "function" or not isinstance(call["id"], str) or not call["id"]:
        raise ProbeFailure("invalid_tool_call")
    if function["name"] != FUNCTION["name"]:
        raise ProbeFailure("unexpected_tool")
    try:
        value = json.loads(function["arguments"])
    except (ValueError, TypeError):
        raise ProbeFailure("invalid_tool_arguments_json") from None
    check_fields(value)


def check_completion(response, check):
    body = response.read(MAX_BYTES + 1)
    if len(body) > MAX_BYTES:
        raise ProbeFailure("response_too_large")
    data = json.loads(body)
    if "error" in data:
        raise ProbeFailure("upstream_error")
    choice = data["choices"][0]
    message = choice["message"]
    if check == "structured_output":
        if message.get("refusal"):
            raise ProbeFailure("model_refusal")
        if choice["finish_reason"] != "stop":
            raise ProbeFailure("unexpected_finish_reason")
        try:
            value = json.loads(message["content"])
        except (TypeError, ValueError):
            raise ProbeFailure("invalid_structured_json") from None
        check_fields(value)
    else:
        if choice["finish_reason"] != "tool_calls":
            raise ProbeFailure("unexpected_finish_reason")
        calls = message["tool_calls"]
        if not isinstance(calls, list) or len(calls) != 1:
            raise ProbeFailure("unexpected_tool_call_count")
        check_tool(calls[0])


def sse_events(response):
    fields = []
    remaining = MAX_BYTES
    while True:
        line = response.readline(remaining + 1)
        if not line:
            return
        remaining -= len(line)
        if remaining < 0:
            raise ProbeFailure("response_too_large")
        line = line.rstrip(b"\r\n")
        if not line:
            if fields:
                yield b"\n".join(fields)
                fields.clear()
        elif line == b"data" or line.startswith(b"data:"):
            value = line[5:]
            fields.append(value[1:] if value.startswith(b" ") else value)


def check_stream(response):
    if response.headers.get_content_type() != "text/event-stream":
        raise ProbeFailure("expected_event_stream")
    calls = {}
    finish_reason = None
    for event in sse_events(response):
        if event == b"[DONE]":
            break
        chunk = json.loads(event)
        if "error" in chunk:
            raise ProbeFailure("upstream_stream_error")
        # Some providers send metadata-only events and a usage-only final chunk.
        for choice in chunk.get("choices", []):
            if choice["index"] != 0:
                raise ProbeFailure("unexpected_choice")
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
            for part in (choice.get("delta") or {}).get("tool_calls") or []:
                call = calls.setdefault(part["index"], {"function": {"arguments": ""}})
                for field in ("type", "id"):
                    if part.get(field) is not None:
                        call[field] = part[field]
                function = part.get("function") or {}
                if function.get("name") is not None:
                    call["function"]["name"] = function["name"]
                call["function"]["arguments"] += function.get("arguments") or ""
    else:
        raise ProbeFailure("stream_missing_done")
    if finish_reason != "tool_calls":
        raise ProbeFailure("unexpected_finish_reason")
    if set(calls) != {0}:
        raise ProbeFailure("unexpected_tool_call_count")
    check_tool(calls[0])


def run_probe(opener, route, check, token):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if check == "streaming_tool_call":
        headers["Accept"] = "text/event-stream"
    data = json.dumps(request_payload(route, check)).encode()
    request = Request(route["url"], data=data, headers=headers, method="POST")
    status = None
    try:
        with opener.open(request, timeout=route["timeout_seconds"]) as response:
            status = response.status
            if status != 200:
                raise ProbeFailure("unexpected_http_status")
            if check == "streaming_tool_call":
                check_stream(response)
            else:
                check_completion(response, check)
    except HTTPError as error:
        error.close()
        return error.code, "unexpected_http_status"
    except ProbeFailure as error:
        return status, str(error)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return status, "invalid_response_format"
    except (OSError, HTTPException):
        return status, "transport_error"
    return status, None


def run_suite(route, samples, token):
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    opener = build_opener(NoRedirects())
    checks = {}
    for check in CHECKS:
        outcomes = [run_probe(opener, route, check, token) for _ in range(samples)]
        checks[check] = {
            "total": samples,
            "passed": sum(reason is None for _, reason in outcomes),
            "http_200": sum(status == 200 for status, _ in outcomes),
            "failures": dict(Counter(reason for _, reason in outcomes if reason)),
        }
    return {"version": VERSION, "started_at": started, "route": route, "checks": checks}


def load_baseline(path, route):
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("version") != VERSION or report.get("route") != route:
        raise ValueError("baseline incompatível com a versão da suíte ou configuração da rota")
    checks = report.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(CHECKS):
        raise ValueError("baseline com testes ausentes ou inválidos")
    for result in checks.values():
        if not isinstance(result, dict):
            raise ValueError("baseline com resultado inválido")
        total, passed = result.get("total"), result.get("passed")
        if type(total) is not int or type(passed) is not int or not 0 <= passed <= total or total < 1:
            raise ValueError("baseline com contagem inválida")
    if not isinstance(report.get("started_at"), str):
        raise ValueError("baseline sem horário de execução")
    return report


def print_summary(report):
    print(f"{report['route']['name']} · {report['route']['model']} · {report['started_at']}")
    for check, result in report["checks"].items():
        print(f"  {check}: {result['passed']}/{result['total']} passaram · "
              f"HTTP 200 em {result['http_200']}/{result['total']}")
        for reason, count in result["failures"].items():
            print(f"    {reason}: {count}")
        if "baseline" in report:
            previous = report["baseline"]["checks"][check]
            delta = 100 * (result["passed"] / result["total"] - previous["passed"] / previous["total"])
            print(f"    baseline: {previous['passed']}/{previous['total']} ({delta:+.1f} p.p.)")


def check_command(args):
    url = urlsplit(args.url)
    if (
        url.scheme not in {"http", "https"} or not url.hostname or not url.path
        or url.username is not None or url.password is not None
        or url.query or url.fragment
        or (url.scheme == "http" and url.hostname not in {"127.0.0.1", "localhost", "::1"})
    ):
        raise ValueError("informe uma URL HTTPS completa, sem credenciais, query ou fragmento")
    if args.samples < 1 or args.timeout < 1:
        raise ValueError("--samples e --timeout devem ser maiores que zero")
    token = os.environ.get(args.key_env) if args.key_env else None
    if args.key_env and not token:
        raise ValueError(f"variável de ambiente ausente ou vazia: {args.key_env}")
    route = {
        "name": args.name, "url": args.url, "model": args.model,
        "upstream": args.openrouter_provider, "timeout_seconds": args.timeout,
    }
    baseline = load_baseline(args.baseline, route) if args.baseline else None
    report = run_suite(route, args.samples, token)
    if baseline:
        report["baseline"] = {
            "started_at": baseline["started_at"],
            "checks": {
                check: {"passed": result["passed"], "total": result["total"]}
                for check, result in baseline["checks"].items()
            },
        }
    if args.output:
        # A failed write must not leave a partial report or overwrite a valid baseline.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=args.output.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            try:
                json.dump(report, temporary, ensure_ascii=False, indent=2)
                temporary.write("\n")
                temporary.close()
                temporary_path.replace(args.output)
            finally:
                temporary_path.unlink(missing_ok=True)
    print_summary(report)
    return int(any(check["passed"] < check["total"] for check in report["checks"].values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=VERSION)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="executar a suíte contra uma rota")
    check.add_argument("--name", required=True, help="nome estável para identificar a rota")
    check.add_argument("--url", required=True, help="URL completa de Chat Completions")
    check.add_argument("--model", required=True)
    check.add_argument("--key-env", help="nome da variável de ambiente com o token")
    check.add_argument("--openrouter-provider", help="slug do provedor a fixar no OpenRouter")
    check.add_argument("--samples", type=int, default=3, help="número de chamadas por teste")
    check.add_argument("--timeout", type=int, default=30, help="timeout de rede em segundos")
    check.add_argument("--baseline", type=Path, help="relatório JSON anterior")
    check.add_argument("--output", type=Path, help="caminho para salvar o relatório JSON")
    args = parser.parse_args()
    try:
        return check_command(args)
    except (OSError, ValueError) as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
