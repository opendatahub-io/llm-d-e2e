"""Configuration types and YAML loaders for test cases and profiles."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import yaml


def parse_duration(s: str) -> timedelta:
    """Parse duration strings like '15m', '2h', '300s', '1h30m'."""
    if not s:
        return timedelta()
    total = timedelta()
    for match in re.finditer(r"(\d+)\s*([hms])", s):
        val, unit = int(match.group(1)), match.group(2)
        if unit == "h":
            total += timedelta(hours=val)
        elif unit == "m":
            total += timedelta(minutes=val)
        elif unit == "s":
            total += timedelta(seconds=val)
    return total


@dataclass
class CacheConfig:
    enabled: bool = True
    pvc_name: str = ""
    storage_size: str = "10Gi"
    storage_class: str = ""
    timeout: timedelta = field(default_factory=lambda: timedelta(minutes=90))
    keep_pvc: bool = True


@dataclass
class ResourceConfig:
    cpu: str = "4"
    memory: str = "32Gi"
    gpus: int = 1
    ephemeral_storage: str = ""
    rdma: bool = False


@dataclass
class ParallelismConfig:
    data: int = 0
    data_local: int = 0
    expert: bool = False
    tensor: int = 0


@dataclass
class PrefillConfig:
    replicas: int = 1
    parallelism: ParallelismConfig | None = None
    resources: ResourceConfig = field(default_factory=ResourceConfig)


@dataclass
class MetricsCheck:
    enabled: bool = False
    check_vllm: bool = False
    check_epp: bool = False
    check_prefix_cache: bool = False
    check_pd: bool = False
    check_scheduler: bool = False
    check_flow_control: bool = False
    check_nixl: bool = False


@dataclass
class MultiPoolCheck:
    enabled: bool = False
    pools: list[dict] = field(default_factory=list)


@dataclass
class BenchmarkThresholds:
    min_output_tokens_per_second: float = 100.0
    max_ttft_median_ms: float = 15000.0
    max_ttft_p95_ms: float = 30000.0
    max_itl_median_ms: float = 50.0
    max_itl_p95_ms: float = 100.0
    max_failed_ratio: float = 0.05


@dataclass
class BenchmarkConfig:
    enabled: bool = False
    image: str = "ghcr.io/vllm-project/guidellm:v0.6.0"
    rate: int = 100
    max_seconds: int = 240
    data: str = (
        "prompt_tokens=8000,prompt_tokens_stdev=8500,prompt_tokens_min=50,prompt_tokens_max=30000,"
        "output_tokens=800,output_tokens_stdev=1500,output_tokens_min=20,output_tokens_max=8000"
    )
    backend_type: str = "openai_http"
    request_type: str = "text_completions"
    warmup_rate: int = 10
    warmup_max_seconds: int = 60
    timeout: timedelta = field(default_factory=lambda: timedelta(minutes=30))
    thresholds: BenchmarkThresholds = field(default_factory=BenchmarkThresholds)


@dataclass
class ChatPrompt:
    role: str = "user"
    content: str = ""


def chat_prompt_to_messages(entry: dict | ChatPrompt) -> list[dict]:
    """Convert a chatPrompts YAML entry (system/user keys) to OpenAI message dicts."""
    if isinstance(entry, dict):
        msgs = []
        if "system" in entry:
            msgs.append({"role": "system", "content": entry["system"]})
        if "user" in entry:
            msgs.append({"role": "user", "content": entry["user"]})
        return msgs
    return [{"role": entry.role, "content": entry.content}]


@dataclass
class ModelConfig:
    name: str = ""
    uri: str = ""
    display_name: str = ""
    category: str = ""
    cache: CacheConfig = field(default_factory=CacheConfig)


@dataclass
class DeployConfig:
    manifest_path: str = ""
    namespace: str = ""
    replicas: int = 1
    service_account: str = ""
    ready_timeout: timedelta = field(default_factory=lambda: timedelta(minutes=15))
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    parallelism: ParallelismConfig | None = None
    prefill: PrefillConfig | None = None
    worker: bool = False
    network_attach: str = ""
    env_overrides: dict[str, str] = field(default_factory=dict)


@dataclass
class ValidateConfig:
    health_endpoint: str = "/health"
    health_port: int = 8000
    health_scheme: str = "HTTPS"
    inference_check: bool = True
    test_prompts: list[str] = field(default_factory=list)
    chat_prompts: list[ChatPrompt] = field(default_factory=list)
    expected_codes: list[int] = field(default_factory=lambda: [200])
    timeout: timedelta = field(default_factory=lambda: timedelta(minutes=2))
    retry_attempts: int = 3
    retry_interval: timedelta = field(default_factory=lambda: timedelta(seconds=15))
    metrics_check: MetricsCheck = field(default_factory=MetricsCheck)
    multi_pool: MultiPoolCheck | None = None
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)


@dataclass
class TestCase:
    name: str = ""
    description: str = ""
    labels: list[str] = field(default_factory=list)
    model: ModelConfig = field(default_factory=ModelConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    validation: ValidateConfig = field(default_factory=ValidateConfig)
    cleanup: bool = True


@dataclass
class TestProfile:
    name: str = ""
    description: str = ""
    platform: str = "any"
    labels: list[str] = field(default_factory=list)
    test_cases: list[str] = field(default_factory=list)
    parallel: bool = False
    timeout: timedelta = field(default_factory=lambda: timedelta(hours=2))


def _snake(key: str) -> str:
    """Convert camelCase YAML keys to snake_case."""
    return re.sub(r"([a-z])([A-Z])", r"\1_\2", key).lower()


def _build(cls, data: dict | None):
    """Recursively build a dataclass from a dict, handling camelCase keys."""
    if not data:
        return cls()
    kwargs = {}
    hints = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    for key, val in data.items():
        snake_key = _snake(key)
        if snake_key not in hints:
            continue
        hint = hints[snake_key]
        if snake_key in ("timeout", "ready_timeout", "retry_interval") and isinstance(val, str):
            kwargs[snake_key] = parse_duration(val)
        elif hint == "CacheConfig" or (isinstance(hint, str) and "CacheConfig" in str(hint)):
            kwargs[snake_key] = _build(CacheConfig, val) if val else CacheConfig()
        elif "ResourceConfig" in str(hint):
            kwargs[snake_key] = _build(ResourceConfig, val) if val else ResourceConfig()
        elif "ParallelismConfig" in str(hint):
            kwargs[snake_key] = _build(ParallelismConfig, val) if val else None
        elif "PrefillConfig" in str(hint):
            kwargs[snake_key] = _build(PrefillConfig, val) if val else None
        elif "MetricsCheck" in str(hint):
            kwargs[snake_key] = _build(MetricsCheck, val) if val else MetricsCheck()
        elif "MultiPoolCheck" in str(hint):
            kwargs[snake_key] = _build(MultiPoolCheck, val) if val else None
        elif "BenchmarkThresholds" in str(hint):
            kwargs[snake_key] = _build(BenchmarkThresholds, val) if val else BenchmarkThresholds()
        elif "BenchmarkConfig" in str(hint):
            kwargs[snake_key] = _build(BenchmarkConfig, val) if val else BenchmarkConfig()
        elif "ModelConfig" in str(hint):
            kwargs[snake_key] = _build(ModelConfig, val) if val else ModelConfig()
        elif "DeployConfig" in str(hint):
            kwargs[snake_key] = _build(DeployConfig, val) if val else DeployConfig()
        elif "ValidateConfig" in str(hint):
            kwargs[snake_key] = _build(ValidateConfig, val) if val else ValidateConfig()
        else:
            kwargs[snake_key] = val
    return cls(**kwargs)


def load_testcase(path: str | Path) -> TestCase:
    with open(path) as f:
        data = yaml.safe_load(f)
    return _build(TestCase, data)


def load_profile(path: str | Path) -> TestProfile:
    with open(path) as f:
        data = yaml.safe_load(f)
    profile = TestProfile(
        name=data.get("name", ""),
        description=data.get("description", ""),
        platform=data.get("platform", "any"),
        labels=data.get("labels", []),
        test_cases=data.get("testCases", []),
        parallel=data.get("parallel", False),
    )
    if "timeout" in data:
        profile.timeout = parse_duration(data["timeout"])
    return profile


def load_testcases_from_dir(directory: str | Path) -> list[TestCase]:
    directory = Path(directory)
    cases = []
    for f in sorted(directory.glob("*.yaml")):
        cases.append(load_testcase(f))
    return cases


def resolve_profile(profile: TestProfile, testcase_dir: str | Path) -> list[TestCase]:
    all_cases = load_testcases_from_dir(testcase_dir)
    by_name = {tc.name: tc for tc in all_cases}
    return [by_name[name] for name in profile.test_cases if name in by_name]


def filter_by_names(cases: list[TestCase], names: list[str]) -> list[TestCase]:
    name_set = set(names)
    return [tc for tc in cases if tc.name in name_set]
