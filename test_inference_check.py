from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import unittest

import inference_check as checker


ROOT = Path(__file__).resolve().parent


class RejectedEndpoint(BaseHTTPRequestHandler):
    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.probes.append(payload)
        self.send_response(429 if payload["stream"] else 401)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format, *args):
        pass


class CustomCaseTests(unittest.TestCase):
    def test_tool_only_case_and_http_codes_through_cli(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), RejectedEndpoint)
        server.probes = []
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join)
        self.addCleanup(server.shutdown)
        with TemporaryDirectory() as directory:
            case_path = Path(directory) / "custom.json"
            report_path = Path(directory) / "report.json"
            spec = json.loads((ROOT / "examples/order.json").read_text(encoding="utf-8"))
            del spec["structured_output"]
            case_path.write_text(json.dumps(spec), encoding="utf-8")
            command = [
                sys.executable, str(ROOT / "inference_check.py"), "check",
                "--url", f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                "--name", "local", "--model", "test", "--samples", "1",
                "--case", str(case_path), "--output", str(report_path),
            ]
            run = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertIn("HTTP 401: 1", run.stdout)
            self.assertIn("HTTP 429: 1", run.stdout)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertNotIn(spec["tool_call"]["prompt"], json.dumps(report))
            self.assertEqual(
                {name: check["http_statuses"] for name, check in report["checks"].items()},
                {"tool_call": {"401": 1}, "streaming_tool_call": {"429": 1}},
            )
            self.assertEqual([probe["stream"] for probe in server.probes], [False, True])
            self.assertEqual(
                [probe["tool_choice"]["function"]["name"] for probe in server.probes],
                ["lookup_order", "lookup_order"],
            )
            self.assertEqual(
                [probe["tools"][0]["function"]["parameters"]["required"] for probe in server.probes],
                [["order_id"], ["order_id"]],
            )
            spec["tool_call"]["prompt"] += " Today."
            case_path.write_text(json.dumps(spec), encoding="utf-8")
            run = subprocess.run(
                [*command, "--baseline", str(report_path)], capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(run.returncode, 2, run.stderr)
            self.assertIn("baseline incompatível com o caso de teste", run.stderr)
            self.assertEqual(len(server.probes), 2)

    def test_nested_objects_and_arrays_preserve_json_types(self):
        spec = deepcopy(checker.DEFAULT_CASE)
        spec["structured_output"]["expected"] = {"values": [True, 1, {"lines": ["x"]}]}
        case = checker.prepare_cases(spec)["structured_output"]
        checker.check_fields({"values": [True, 1, {"lines": ["x"]}]}, case)
        with self.assertRaisesRegex(checker.ProbeFailure, "unexpected_values"):
            checker.check_fields({"values": [1, True, {"lines": ["x"]}]}, case)
        with self.assertRaisesRegex(checker.ProbeFailure, "schema_mismatch"):
            checker.check_fields({"values": [True, 1, {"lines": [3]}]}, case)

    def test_thresholds_are_inclusive_and_independent(self):
        report = {
            "checks": {"tool_call": {"passed": 3, "total": 4}},
            "baseline": {"checks": {"tool_call": {"passed": 4, "total": 5}}},
        }
        self.assertEqual(checker.evaluate_gate(report, 75, 5)["violations"], [])
        self.assertEqual(checker.evaluate_gate(report, 74, 4.9)["violations"], [
            {"check": "tool_call", "reason": "max_baseline_drop"},
        ])
        self.assertEqual(checker.evaluate_gate(report, 75.1, 5)["violations"], [
            {"check": "tool_call", "reason": "min_pass_rate"},
        ])

    def test_default_case_accepts_v01_baseline(self):
        route = {"name": "test"}
        report = {
            "version": "0.1.0", "started_at": "2026-11-21T00:00:00+00:00", "route": route,
            "checks": {check: {"total": 3, "passed": 3} for check in checker.DEFAULT_CASES},
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(checker.load_baseline(path, route, checker.DEFAULT_CASES, None), report)


if __name__ == "__main__":
    unittest.main()
