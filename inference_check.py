"""Run synthetic Chat Completions compatibility probes outside the app's request path."""

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
from http.client import HTTPException
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

VERSION = "0.2.0"
DEFAULT_CASE = {
    "structured_output": {
        "prompt": 'Return city "Recife" and days 2 in the requested JSON schema.',
        "expected": {"city": "Recife", "days": 2},
    },
    "tool_call": {
        "prompt": 'Call get_forecast for "Recife" with days set to 2.',
        "name": "get_forecast",
        "description": "Get the forecast for a city and number of days.",
        "expected_arguments": {"city": "Recife", "days": 2},
    },
}
MAX_BYTES = 1_048_576
JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean", type(None): "null"}


class ProbeFailure(Exception):
    """A diagnostic that can be recorded without copying upstream content."""


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def example_schema(value):
    if isinstance(value, dict):
        return {
            "type": "object",
            "properties": {key: example_schema(item) for key, item in value.items()},
            "required": list(value),
            "additionalProperties": False,
        }
    if isinstance(value, list):
        schemas = {}
        for item in value:
            schema = example_schema(item)
            schemas[canonical_json(schema)] = schema
        items = list(schemas.values())
        if len(items) > 1:
            return {"type": "array", "items": {"anyOf": items}}
        return {"type": "array", "items": items[0] if items else {"type": "null"}}
    return {"type": JSON_TYPES[type(value)]}


def matches_schema(value, schema):
    if "anyOf" in schema:
        return any(matches_schema(value, option) for option in schema["anyOf"])
    kind = schema["type"]
    if kind == "object":
        properties = schema["properties"]
        return isinstance(value, dict) and value.keys() == properties.keys() and all(
            matches_schema(value[key], field) for key, field in properties.items()
        )
    if kind == "array":
        return isinstance(value, list) and all(matches_schema(item, schema["items"]) for item in value)
    return JSON_TYPES.get(type(value)) == kind or (kind == "number" and type(value) is int)


@dataclass(frozen=True)
class ProbeCase:
    prompt: str
    expected: dict
    schema: dict
    name: str
    description: str = ""


def prepare_cases(spec, structured_name="expected_response"):
    if not isinstance(spec, dict) or not spec or spec.keys() - DEFAULT_CASE.keys():
        raise ValueError("o caso deve conter structured_output, tool_call ou ambos")
    cases = {}
    for check in DEFAULT_CASE:
        if check not in spec:
            continue
        config = spec[check]
        if not isinstance(config, dict) or config.keys() != DEFAULT_CASE[check].keys():
            fields = ", ".join(DEFAULT_CASE[check])
            raise ValueError(f"{check} deve conter os campos {fields}")
        prompt = config["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{check}.prompt deve ser um texto não vazio")
        structured = check == "structured_output"
        name = structured_name if structured else config["name"]
        description = "" if structured else config["description"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name):
            raise ValueError("tool_call.name deve ter de 1 a 64 letras ASCII, números, hífens ou sublinhados")
        if not isinstance(description, str):
            raise ValueError("tool_call.description deve ser um texto")
        expected = config["expected"] if structured else config["expected_arguments"]
        if not isinstance(expected, dict):
            raise ValueError(f"a resposta esperada de {check} deve ser um objeto JSON")
        cases[check] = ProbeCase(prompt, expected, example_schema(expected), name, description)
    if "tool_call" in cases:
        cases["streaming_tool_call"] = cases["tool_call"]
    return cases


DEFAULT_CASES = prepare_cases(DEFAULT_CASE, structured_name="forecast_request")


def load_cases(path):
    if path is None:
        return DEFAULT_CASES, None
    try:
        encoded = canonical_json(json.loads(path.read_text(encoding="utf-8")))
        cases = prepare_cases(json.loads(encoded))
    except (RecursionError, OverflowError) as error:
        raise ValueError("o caso JSON é complexo demais") from error
    return cases, sha256(encoded.encode()).hexdigest()


def request_payload(route, check, case=None):
    if case is None:
        case = DEFAULT_CASES[check]
    structured = check == "structured_output"
    payload = {
        "model": route["model"],
        "messages": [{"role": "user", "content": case.prompt}],
        "stream": check == "streaming_tool_call",
    }
    if structured:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": case.name, "schema": case.schema, "strict": True},
        }
    else:
        function = {
            "name": case.name, "description": case.description, "parameters": case.schema, "strict": True,
        }
        payload["tools"] = [{"type": "function", "function": function}]
        payload["tool_choice"] = {"type": "function", "function": {"name": case.name}}
    if route["upstream"]:
        payload["provider"] = {
            "only": [route["upstream"]],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
    return payload


def check_fields(value, case):
    if not matches_schema(value, case.schema):
        raise ProbeFailure("schema_mismatch")
    if canonical_json(value) != canonical_json(case.expected):
        raise ProbeFailure("unexpected_values")


def check_tool(call, case):
    function = call["function"]
    if call["type"] != "function" or not isinstance(call["id"], str) or not call["id"]:
        raise ProbeFailure("invalid_tool_call")
    if function["name"] != case.name:
        raise ProbeFailure("unexpected_tool")
    try:
        value = json.loads(function["arguments"])
    except (ValueError, TypeError):
        raise ProbeFailure("invalid_tool_arguments_json") from None
    check_fields(value, case)


def check_completion(response, check, case):
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
        check_fields(value, case)
    else:
        if choice["finish_reason"] != "tool_calls":
            raise ProbeFailure("unexpected_finish_reason")
        calls = message["tool_calls"]
        if not isinstance(calls, list) or len(calls) != 1:
            raise ProbeFailure("unexpected_tool_call_count")
        check_tool(calls[0], case)


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


def check_stream(response, case):
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
    check_tool(calls[0], case)


def run_probe(opener, route, check, token, case):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if check == "streaming_tool_call":
        headers["Accept"] = "text/event-stream"
    data = json.dumps(request_payload(route, check, case)).encode()
    request = Request(route["url"], data=data, headers=headers, method="POST")
    status = None
    try:
        with opener.open(request, timeout=route["timeout_seconds"]) as response:
            status = response.status
            if status != 200:
                raise ProbeFailure("unexpected_http_status")
            if check == "streaming_tool_call":
                check_stream(response, case)
            else:
                check_completion(response, check, case)
    except HTTPError as error:
        error.close()
        return error.code, "unexpected_http_status"
    except ProbeFailure as error:
        return status, str(error)
    except (AttributeError, IndexError, KeyError, RecursionError, TypeError, ValueError):
        return status, "invalid_response_format"
    except (OSError, HTTPException):
        return status, "transport_error"
    return status, None


def run_suite(route, samples, token, cases=DEFAULT_CASES, case_sha256=None):
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    opener = build_opener(NoRedirects())
    checks = {}
    for check, case in cases.items():
        outcomes = [run_probe(opener, route, check, token, case) for _ in range(samples)]
        statuses = Counter(status for status, _ in outcomes if status is not None)
        checks[check] = {
            "total": samples,
            "passed": sum(reason is None for _, reason in outcomes),
            "http_200": statuses[200],
            "http_statuses": {str(status): statuses[status] for status in sorted(statuses)},
            "failures": dict(Counter(reason for _, reason in outcomes if reason)),
        }
    return {
        "version": VERSION, "started_at": started, "route": route,
        "case_sha256": case_sha256, "checks": checks,
    }


def load_baseline(path, route, cases, case_sha256):
    report = json.loads(path.read_text(encoding="utf-8"))
    supported_versions = (VERSION,) if case_sha256 is not None else (VERSION, "0.1.0")
    if not isinstance(report, dict) or report.get("version") not in supported_versions or report.get("route") != route:
        raise ValueError("baseline incompatível com a versão da suíte ou configuração da rota")
    if report.get("case_sha256") != case_sha256:
        raise ValueError("baseline incompatível com o caso de teste")
    checks = report.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(cases):
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
        for status, count in result["http_statuses"].items():
            if status != "200":
                print(f"    HTTP {status}: {count}")
        if "baseline" in report:
            previous = report["baseline"]["checks"][check]
            delta = 100 * (result["passed"] / result["total"] - previous["passed"] / previous["total"])
            print(f"    baseline: {previous['passed']}/{previous['total']} ({delta:+.1f} p.p.)")
    for violation in report.get("gate", {}).get("violations", []):
        print(f"  Limite de CI excedido em {violation['check']} ({violation['reason']}).")


def evaluate_gate(report, minimum, max_drop):
    violations = []
    for check, result in report["checks"].items():
        rate = Fraction(result["passed"] * 100, result["total"])
        if rate < Fraction(str(minimum)):
            violations.append({"check": check, "reason": "min_pass_rate"})
        if max_drop is not None:
            previous = report["baseline"]["checks"][check]
            previous_rate = Fraction(previous["passed"] * 100, previous["total"])
            if previous_rate - rate > Fraction(str(max_drop)):
                violations.append({"check": check, "reason": "max_baseline_drop"})
    return {"min_pass_rate": minimum, "max_baseline_drop": max_drop, "violations": violations}


def percentage(text):
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError("esperado um número entre 0 e 100") from None
    if not math.isfinite(value) or not 0 <= value <= 100:
        raise argparse.ArgumentTypeError("esperado um número entre 0 e 100")
    return value


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
    if args.max_baseline_drop is not None and args.baseline is None:
        raise ValueError("--max-baseline-drop requer --baseline")
    token = os.environ.get(args.key_env) if args.key_env else None
    if args.key_env and not token:
        raise ValueError(f"variável de ambiente ausente ou vazia: {args.key_env}")
    route = {
        "name": args.name, "url": args.url, "model": args.model,
        "upstream": args.openrouter_provider, "timeout_seconds": args.timeout,
    }
    cases, case_sha256 = load_cases(args.case)
    baseline = load_baseline(args.baseline, route, cases, case_sha256) if args.baseline else None
    report = run_suite(route, args.samples, token, cases, case_sha256)
    if baseline:
        report["baseline"] = {
            "started_at": baseline["started_at"],
            "checks": {
                check: {"passed": result["passed"], "total": result["total"]}
                for check, result in baseline["checks"].items()
            },
        }
    report["gate"] = evaluate_gate(report, args.min_pass_rate, args.max_baseline_drop)
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
    return int(bool(report["gate"]["violations"]))


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
    check.add_argument("--case", type=Path, help="arquivo JSON com prompts e resultados esperados")
    check.add_argument("--baseline", type=Path, help="relatório JSON anterior")
    check.add_argument("--min-pass-rate", type=percentage, default=100.0,
                       help="percentual mínimo de aprovação por teste, padrão 100")
    check.add_argument("--max-baseline-drop", type=percentage,
                       help="queda máxima permitida por teste em pontos percentuais")
    check.add_argument("--output", type=Path, help="caminho para salvar o relatório JSON")
    args = parser.parse_args()
    try:
        return check_command(args)
    except (OSError, ValueError) as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
