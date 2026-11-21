"""Reproduce an HTTP 200 regression and check the CLI's exit status and report."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
from threading import Thread


class DemoEndpoint(BaseHTTPRequestHandler):
    streaming_calls = 0
    degraded = False

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        tool = {"id": "call_demo", "type": "function", "function": {
            "name": "get_forecast", "arguments": '{"city":"Recife","days":2}',
        }}
        if payload["stream"]:
            type(self).streaming_calls += 1
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            first_part = {"index": 0, **tool, "function": {
                "name": "get_forecast", "arguments": '{"city":"Recife",',
            }}
            self.send_chunk({"tool_calls": [first_part]})
            if not self.degraded or self.streaming_calls % 5 not in {1, 2}:
                self.send_chunk({"tool_calls": [
                    {"index": 0, "function": {"arguments": '"days":2}'}},
                ]})
            self.send_chunk({}, "tool_calls")
            self.wfile.write(b"data: [DONE]\n\n")
            return
        if "response_format" in payload:
            message = {"content": '{"city":"Recife","days":2}'}
            finish = "stop"
        else:
            message = {"tool_calls": [tool]}
            finish = "tool_calls"
        body = json.dumps({"choices": [{"message": message, "finish_reason": finish}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_chunk(self, delta, finish=None):
        chunk = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        self.wfile.write(f"data: {json.dumps(chunk)}\r\n\r\n".encode())
        self.wfile.flush()

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("demo-reports"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = args.output_dir / "baseline.json"
    current = args.output_dir / "regression.json"

    with ThreadingHTTPServer(("127.0.0.1", 0), DemoEndpoint) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            command = [
                sys.executable, str(Path(__file__).with_name("inference_check.py")), "check",
                "--url", f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                "--name", "demo-local", "--model", "demo-model", "--samples", "10",
            ]
            print("Execução inicial", flush=True)
            subprocess.run([*command, "--output", str(baseline)], check=True)
            DemoEndpoint.degraded = True
            print("\nExecução com regressão", flush=True)
            run = subprocess.run([*command, "--baseline", str(baseline), "--output", str(current)])
            if run.returncode != 1:
                sys.exit(f"Demo falhou. Esperava exit code 1, recebi {run.returncode}.")
        finally:
            server.shutdown()
            thread.join()

    report = json.loads(current.read_text(encoding="utf-8"))
    if report["checks"]["streaming_tool_call"] != {
        "total": 10, "passed": 6, "http_200": 10, "failures": {"invalid_tool_arguments_json": 4},
    }:
        sys.exit("Demo falhou. Esperava exatamente quatro streams com argumentos incompletos.")
    if any(report["checks"][check]["passed"] != 10 for check in ("structured_output", "tool_call")):
        sys.exit("Demo falhou. Os outros contratos deveriam passar em todas as chamadas.")
    print(f"\nRegressão detectada com exit code 1. Relatórios em {args.output_dir}.")


if __name__ == "__main__":
    main()
