"""JSON conformance report written at session end.

Session-scoped ``report`` fixture constructs ``Report`` with profile/platform;
``finalize(--report-dir)`` writes ``report-<timestamp>.json`` including:
  - suite metadata and ``manifests`` from ``deploy/manifests/.manifest-ref``
    (branch, repo, commit, date — via ``load_manifest_ref``)
  - per-case ``TestResult`` rows (pass/fail/skip, duration, error, model)
  - summary counts (total / passed / failed / skipped)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class TestResult:
    name: str
    description: str = ""
    category: str = ""
    status: str = "skip"  # pass, fail, skip
    duration: float = 0.0
    error: str = ""
    model: str = ""


def load_manifest_ref(manifest_dir: str = "deploy/manifests") -> dict:
    ref_file = Path(manifest_dir) / ".manifest-ref"
    if not ref_file.exists():
        return {}
    info = {}
    for line in ref_file.read_text().splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            info[k.strip()] = v.strip()
    return info


@dataclass
class Report:
    suite: str = "llm-d-e2e"
    profile: str = ""
    platform: str = ""
    start_time: str = ""
    results: list[TestResult] = field(default_factory=list)

    def add(self, result: TestResult):
        self.results.append(result)

    def finalize(self, report_dir: str = "reports") -> str:
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        end_time = datetime.utcnow().isoformat()
        data = {
            "suite": self.suite,
            "profile": self.profile,
            "platform": self.platform,
            "manifests": load_manifest_ref(),
            "start_time": self.start_time,
            "end_time": end_time,
            "results": [asdict(r) for r in self.results],
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results if r.status == "pass"),
                "failed": sum(1 for r in self.results if r.status == "fail"),
                "skipped": sum(1 for r in self.results if r.status == "skip"),
            },
        }
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        path = Path(report_dir) / f"report-{ts}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        log.info("Report written to %s", path)
        return str(path)
