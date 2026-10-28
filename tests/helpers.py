"""A factory config with paid roles, for tests of the client, tiers, batches, and agents.

The shipped config keeps every role on ARC until a run escalates, so the Claude and OpenAI code paths are tested
against this one: the shipped config plus roles that call paid models directly.
"""

from pathlib import Path

import yaml

from factory.config import FactoryConfig, load_config

ROOT = Path(__file__).resolve().parents[1]
PAID_ROLES = {
    "writer": {"provider": "arc", "model": "DeepSeek-V4.1-Flash"},
    "gen_agent": {"provider": "claude", "model": "claude-sonnet-5-5", "effort": "high",
                  "quality": {"model": "claude-opus-5-5"}},
    "grader": {"provider": "arc", "model": "gpt-oss-120b",
               "quality": {"provider": "openai", "model": "gpt-6.1-sol", "max_tokens": 16000}},
    "agent": {"provider": "claude", "model": "claude-sonnet-5-5", "effort": "high",
              "quality": {"model": "claude-opus-5-5"}},
}


def paid_config_path(tmp_path: Path, **overrides) -> Path:
    data = yaml.safe_load((ROOT / "configs" / "factory.yaml").read_text())
    data["roles"].update(PAID_ROLES)
    data["budget"] = {"max_real_usd": 10.0}
    data["credits"] = {}
    data["models_file"] = str(ROOT / "configs" / "models.yaml")
    data.update(overrides)
    path = tmp_path / "paid" / "configs" / "factory.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))
    return path


def paid_config(tmp_path: Path, tier: str | None = None, **overrides) -> FactoryConfig:
    return load_config(paid_config_path(tmp_path, **overrides), tier=tier)
