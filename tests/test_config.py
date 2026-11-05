"""The factory config loads, starts every role on ARC, and enforces family separation in both tiers."""

import os
from pathlib import Path

import pytest
import yaml

from factory.config import load_config
from tests.helpers import paid_config_path

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return load_config(ROOT / "configs" / "factory.yaml")


def write(tmp_path, data) -> Path:
    path = tmp_path / "configs" / "factory.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    data["models_file"] = str(ROOT / "configs" / "models.yaml")
    path.write_text(yaml.safe_dump(data))
    return path


def shipped():
    return yaml.safe_load((ROOT / "configs" / "factory.yaml").read_text())


def test_factory_config_loads_and_resolves_paths(cfg):
    assert cfg.root == ROOT
    assert cfg.providers["claude"].data_scope == "public_only"
    assert cfg.providers["arc"].api_key_env == "ARC_API_KEY"


def test_every_role_starts_on_a_free_arc_model(cfg):
    for name, role in cfg.roles.items():
        assert role.provider == "arc" and not cfg.model_info(role.model).real, name


def test_paid_models_are_escalation_steps_only(cfg):
    assert cfg.escalations["coder"].model == "gpt-6.1-sol"
    assert cfg.escalations["reviewer"].model == "claude-sonnet-5-5"
    assert cfg.model_info(cfg.escalations["reviewer"].model).real


def test_shipped_roles_respect_family_separation_in_both_tiers(cfg):
    assert cfg.separation_errors() == []
    assert cfg.families("coder") == {"glm", "openai"}
    assert cfg.families("reviewer") == {"deepseek", "claude"}


def test_a_reviewer_sharing_an_authors_family_is_rejected(tmp_path):
    data = shipped()
    data["roles"]["reviewer"]["model"] = "GLM-5.3"  # GLM writes the code
    with pytest.raises(ValueError, match="family separation"):
        load_config(write(tmp_path, data))


def test_escalation_steps_count_for_separation(tmp_path):
    data = shipped()
    data["roles"]["coder"]["quality"] = {"provider": "claude", "model": "claude-opus-5-5"}  # Claude also reviews
    with pytest.raises(ValueError, match="reviewer 'reviewer' shares \\['claude'\\] with author 'coder'"):
        load_config(write(tmp_path, data))


def test_unknown_model_fails_at_load(tmp_path):
    data = {"name": "t", "providers": {"p": {"format": "openai", "base_url": "http://x"}},
            "roles": {"r": {"provider": "p", "model": "no-such-model"}}}
    with pytest.raises(KeyError, match="no-such-model"):
        load_config(write(tmp_path, data))


def test_credits_must_name_known_providers(tmp_path):
    data = shipped()
    data["credits"]["by_provider"]["gemini"] = 5
    with pytest.raises(ValueError, match="gemini"):
        load_config(write(tmp_path, data))


def test_tiers_switch_roles_to_their_escalation_step(monkeypatch, tmp_path):
    monkeypatch.delenv("FACTORY_TIER", raising=False)
    path = paid_config_path(tmp_path)
    eco = load_config(path, tier="economy")
    q = load_config(path, tier="quality")
    assert eco.roles["agent"].model == "claude-sonnet-5-5" and q.roles["agent"].model == "claude-opus-5-5"
    assert eco.roles["coder"].model == "GLM-5.3-thinking-high" and q.roles["coder"].model == "gpt-6.1-sol"
    assert q.roles["coder"].provider == "openai"
    monkeypatch.setenv("FACTORY_TIER", "quality")
    assert load_config(path).roles["agent"].model == "claude-opus-5-5"


def test_one_role_can_take_its_escalation_step(monkeypatch, tmp_path):
    monkeypatch.delenv("FACTORY_TIER", raising=False)
    monkeypatch.setenv("FACTORY_ROLE_TIERS", "reviewer=quality")
    cfg = load_config(paid_config_path(tmp_path))
    assert cfg.roles["reviewer"].model == "claude-sonnet-5-5"
    assert cfg.roles["coder"].model == "GLM-5.3-thinking-high"
    monkeypatch.setenv("FACTORY_ROLE_TIERS", "nobody=quality")
    with pytest.raises(ValueError, match="unknown roles"):
        load_config(paid_config_path(tmp_path))


def test_a_role_cannot_escalate_from_one_free_model_to_another(tmp_path):
    data = shipped()
    data["roles"]["planner"]["quality"] = {"model": "GLM-5.3"}
    with pytest.raises(ValueError, match="both tiers are free"):
        load_config(write(tmp_path, data))


@pytest.fixture(autouse=True)
def _restore_environ():
    """load_dotenv writes the real os.environ; put back exactly what was there, whatever keys a test's .env set."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def clear_factory_env(monkeypatch):
    for key in ("ASPIRE_DATA_PRIVATE", "FACTORY_TIER", "FACTORY_ROLE_TIERS", "ARC_API_KEY"):
        monkeypatch.delenv(key, raising=False)


def test_dotenv_fills_missing_keys_but_never_overrides_the_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("ARC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    (tmp_path / ".env").write_text("# keys\nARC_API_KEY='from-file'\nOPENAI_API_KEY=from-file\n")
    load_config(write(tmp_path, shipped()))
    import os
    assert os.environ["ARC_API_KEY"] == "from-file" and os.environ["OPENAI_API_KEY"] == "from-env"
    monkeypatch.delenv("ARC_API_KEY")


def test_aspire_data_private_makes_lab_data_unsendable_to_paid_providers(monkeypatch):
    monkeypatch.setenv("ASPIRE_DATA_PRIVATE", "1")
    assert load_config(ROOT / "configs" / "factory.yaml").data_public is False
    monkeypatch.setenv("ASPIRE_DATA_PRIVATE", "0")
    assert load_config(ROOT / "configs" / "factory.yaml").data_public is True


@pytest.mark.parametrize("line", ["ASPIRE_DATA_PRIVATE=1", "export ASPIRE_DATA_PRIVATE=1", "ASPIRE_DATA_PRIVATE='1'"])
def test_aspire_data_private_set_only_in_dotenv_still_blocks_paid_providers(monkeypatch, tmp_path, line):
    clear_factory_env(monkeypatch)
    (tmp_path / ".env").write_text(f"# flags\n{line}\n")
    assert load_config(write(tmp_path, {**shipped(), "data_public": True})).data_public is False


@pytest.mark.parametrize("shell", ["", "0"])
def test_dotenv_private_flag_beats_an_empty_or_zero_shell_value(monkeypatch, tmp_path, shell):
    monkeypatch.setenv("ASPIRE_DATA_PRIVATE", shell)
    (tmp_path / ".env").write_text("ASPIRE_DATA_PRIVATE=1\n")
    assert load_config(write(tmp_path, {**shipped(), "data_public": True})).data_public is False
    assert os.environ["ASPIRE_DATA_PRIVATE"] == "1"  # os.environ agrees with the config


@pytest.mark.parametrize("raw", [b"\xef\xbb\xbfASPIRE_DATA_PRIVATE=1\n", b"export\tASPIRE_DATA_PRIVATE=1\n"])
def test_dotenv_parses_bom_and_tab_after_export(monkeypatch, tmp_path, raw):
    clear_factory_env(monkeypatch)
    (tmp_path / ".env").write_bytes(raw)
    assert load_config(write(tmp_path, {**shipped(), "data_public": True})).data_public is False


def test_dotenv_duplicate_keys_agree_between_environ_and_the_guard(monkeypatch, tmp_path):
    clear_factory_env(monkeypatch)
    (tmp_path / ".env").write_text("ASPIRE_DATA_PRIVATE=1\nASPIRE_DATA_PRIVATE=0\n")
    assert load_config(write(tmp_path, {**shipped(), "data_public": True})).data_public is False
    assert os.environ["ASPIRE_DATA_PRIVATE"] == "1"


def test_dotenv_private_flag_on_any_line_counts(monkeypatch, tmp_path):
    clear_factory_env(monkeypatch)
    (tmp_path / ".env").write_text("ASPIRE_DATA_PRIVATE=0\nASPIRE_DATA_PRIVATE=1\n")
    assert load_config(write(tmp_path, {**shipped(), "data_public": True})).data_public is False
    assert os.environ["ASPIRE_DATA_PRIVATE"] == "1"


def test_dotenv_cannot_loosen_a_shell_set_guard(monkeypatch, tmp_path):
    monkeypatch.setenv("ASPIRE_DATA_PRIVATE", "1")
    (tmp_path / ".env").write_text("ASPIRE_DATA_PRIVATE=0\n")
    assert load_config(write(tmp_path, {**shipped(), "data_public": True})).data_public is False


def test_dotenv_malformed_quote_on_the_lab_flag_fails_closed(monkeypatch, tmp_path):
    clear_factory_env(monkeypatch)
    (tmp_path / ".env").write_text('ASPIRE_DATA_PRIVATE="0\n')
    assert load_config(write(tmp_path, {**shipped(), "data_public": True})).data_public is False


@pytest.mark.parametrize("line", ['ASPIRE_DATA_PRIVATE="0"1', "ASPIRE_DATA_PRIVATE='0'x"])
def test_dotenv_text_after_a_closing_quote_fails_closed(monkeypatch, tmp_path, line):
    clear_factory_env(monkeypatch)
    (tmp_path / ".env").write_text(line + "\n")
    assert load_config(write(tmp_path, {**shipped(), "data_public": True})).data_public is False


def test_dotenv_other_keys_still_lose_stray_quotes(monkeypatch, tmp_path):
    clear_factory_env(monkeypatch)
    (tmp_path / ".env").write_text("ARC_API_KEY=abc'\n")
    load_config(write(tmp_path, shipped()))
    assert os.environ["ARC_API_KEY"] == "abc"


def test_dotenv_unquoted_inline_comment_fails_closed(monkeypatch, tmp_path):
    clear_factory_env(monkeypatch)
    (tmp_path / ".env").write_text("ASPIRE_DATA_PRIVATE=0  # off\nARC_API_KEY=ab #cd\n")
    assert load_config(write(tmp_path, {**shipped(), "data_public": True})).data_public is False
    assert os.environ["ARC_API_KEY"] == "ab #cd"  # values are never truncated


def test_dotenv_quoted_values_with_trailing_comments(monkeypatch, tmp_path):
    clear_factory_env(monkeypatch)
    (tmp_path / ".env").write_text('ASPIRE_DATA_PRIVATE="0" # off\nARC_API_KEY="a #b"\n')
    cfg = load_config(write(tmp_path, {**shipped(), "data_public": True}))
    assert cfg.data_public is True and os.environ["ARC_API_KEY"] == "a #b"


def test_spend_tier_settings_in_dotenv_are_ignored(monkeypatch, tmp_path):
    clear_factory_env(monkeypatch)
    (tmp_path / ".env").write_text("FACTORY_TIER=quality\nFACTORY_ROLE_TIERS=planner=quality\n")
    cfg = load_config(write(tmp_path, shipped()))
    assert cfg.tier == "economy" and cfg.role_tiers == {}  # only the real environment can pick a paid tier
    again = load_config(write(tmp_path, shipped()))  # a second load in the same process must not pick them up either
    assert again.tier == "economy" and again.role_tiers == {}


def test_mcf7_baseline_refuses_to_run_with_public_data(monkeypatch, tmp_path):
    import importlib.util

    import factory.llm as llm_mod
    spec = importlib.util.spec_from_file_location("baseline", ROOT / "bench/research/mcf7_release/baseline.py")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    seen = {}

    class Fake:
        def __init__(self, cfg, run_dir, **kw):
            seen["public"] = cfg.data_public

        def complete(self, role, user, *a, **kw):
            seen["role"], seen["data"] = role, kw["data"]
            raise SystemExit(0)

    monkeypatch.setenv("ASPIRE_DATA_PRIVATE", "0")  # restored after the test; the script must force it on
    monkeypatch.setattr(llm_mod, "LLM", Fake)
    lst = tmp_path / "list.csv"
    lst.write_text("accession,gene\nP06493,CDK1\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        baseline.main([str(lst), "--out", str(tmp_path / "out")])
    assert seen == {"public": False, "role": "agent", "data": "private"}
