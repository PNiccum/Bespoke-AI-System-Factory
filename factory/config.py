"""Factory configuration.

One YAML file under ``configs/`` says which providers exist, which model plays each role, and what a run may
spend. The code holds no model names; swapping a model is a config change.

The model catalog (``configs/models.yaml``) gives each model a family and a price. Free ARC models carry a
shadow price so every run is costed as if every model were paid.

Lifted from Bespoke_LLM's lab config; the corpus, expansion and training sections were dropped.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------------------
# Models, providers, roles
# --------------------------------------------------------------------------

class ModelInfo(BaseModel):
    """One entry of the model catalog. Prices are USD per million tokens."""

    family: str  # used for family separation: no family reviews what it wrote
    input: float = 0.0
    output: float = 0.0
    cache_read: float | None = None
    cache_write: float | None = None
    batch_discount: float = 0.5  # multiplier for Message Batches calls
    real: bool = True  # False for free models; they are costed at shadow prices
    shadow_of: str | None = None  # catalog model whose prices a free model borrows


class ProviderConfig(BaseModel):
    format: Literal["anthropic", "openai"]
    base_url: str | None = None  # None = the SDK default (Claude API)
    api_key_env: str | None = None
    api_key_file: Path | None = None
    max_concurrency: int = 8
    timeout_s: float = 1800
    stream_retries: int = 4
    # "public_only": the provider may only receive data the config declares public (data_public). Calls that
    # are not marked data="public" are refused before anything leaves the machine.
    data_scope: Literal["any", "public_only"] = "any"
    # OpenAI's reasoning models take max_completion_tokens and reject max_tokens.
    token_param: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"  # noqa: S105 (a parameter name)
    # Anthropic prompt caching: mark the system prompt, the tool list, and the newest message as cache
    # breakpoints, so each agent turn reads the earlier history at the cache rate. Claude API only: other
    # Anthropic-format gateways may reject the markers. OpenAI caches long prompts automatically.
    prompt_caching: bool = False
    # The provider has a batch API (Claude Message Batches, OpenAI Batch): half price, results within 24 h.
    # Without one, LLM.complete_batch makes concurrent single calls instead.
    batch: bool = False


class RoleConfig(BaseModel):
    provider: str
    model: str
    # "author" roles produce work; "reviewer" roles judge it. A reviewer never shares a model family with an
    # author, in either tier, so no model grades its own output.
    kind: Literal["author", "reviewer", "other"] = "other"
    max_tokens: int = 32000
    effort: str | None = None  # Anthropic output_config.effort; ignored by the ARC gateway
    params: dict = {}  # extra provider-specific request fields
    # The escalation step for this role, used when the tier is "quality": any of provider, model, max_tokens,
    # effort, params. The role's own fields are the free (ARC) default.
    quality: dict | None = None
    # Free models on the same provider to try, in order, when this one is unavailable (not found, or still failing
    # after the retries). They must be free and count toward family separation; the paid step never falls back.
    fallbacks: list[str] = []
    # A faster free model of the same family (one of the fallbacks) for the retry after a reply hit max_tokens:
    # a thinking model that overran its budget once usually overruns it again on the same request.
    on_truncation: str | None = None


class SandboxConfig(BaseModel):
    """Where generated code's tests run (factory/tools/sandbox.py). Exported as ASPIRE_SANDBOX* environment variables,
    which the MCP server's process inherits; a variable already set in the environment wins, for one-off runs."""
    runner: Literal["process", "docker", "podman", "apptainer"] = "process"
    image: str = "aspire-sandbox"  # a .sif path for apptainer
    memory: str = "2g"
    cpus: str = "2"
    pids: str = "256"

    def export(self) -> None:
        os.environ.setdefault("ASPIRE_SANDBOX", self.runner)
        os.environ.setdefault("ASPIRE_SANDBOX_IMAGE", self.image)
        os.environ.setdefault("ASPIRE_SANDBOX_LIMITS", f"{self.memory},{self.cpus},{self.pids}")


class Budget(BaseModel):
    """Caps for one run. Dollars are REAL spend only: free ARC calls count as zero, so a free run is never stopped
    by a cost that does not exist. Shadow cost is reported, never enforced."""

    max_real_usd: float | None = None  # hard stop for the run
    # Soft limit: a paid call that would take the run past this needs a person's approval (factory/llm/gate.py).
    approve_above_usd: float | None = None
    max_tokens: int | None = None
    max_seconds: float | None = None
    # Token ceiling for one entry into a pipeline stage (keys: planning, research, production, critic, qa, redesign),
    # counting every model call made inside it. Checked before each call, so a stage can overshoot by the calls
    # already in flight. Reaching it ends the stage with StageCapReached (factory/llm/cost.py), not the run.
    stage_max_tokens: dict[str, int] = {}


class Credits(BaseModel):
    """What is left on each paid account, tracked across runs in the workspace's spend ledger. Paid calls stop
    when a provider's credit is used up, or when total spend would eat into the reserve held back for the demo."""

    by_provider: dict[str, float] = {}  # provider name -> USD available, e.g. {claude: 20, openai: 15}
    reserve_usd: float = 0.0


class AgentCaps(BaseModel):
    """Caps for one agent run. The dollar cap is real spend only; an agent on a free model is bounded by its step,
    token, and time caps."""

    max_steps: int = 40
    max_tokens: int = 2_000_000
    max_seconds: float = 3600
    max_real_usd: float = 1.0
    # Context compaction: every `compact_every` turns, tool results older than the last `keep_recent`
    # turns are replaced by one-line stubs. Done at intervals, not every turn, so the history between
    # compactions stays a stable prefix for prompt caching. 0 turns it off.
    compact_every: int = 8
    keep_recent: int = 4


class FactoryConfig(BaseModel):
    name: str
    description: str = ""
    workspace: Path = Path("workspace")
    runs_dir: Path = Path("runs")
    registry_dir: Path = Path("registry")
    models_file: Path = Path("configs/models.yaml")
    # True when every input the factory is given may go to a public_only provider (Claude, OpenAI).
    # The class demo uses public example data only.
    data_public: bool = False
    providers: dict[str, ProviderConfig]
    roles: dict[str, RoleConfig]
    budget: Budget = Field(default_factory=Budget)
    credits: Credits = Field(default_factory=Credits)
    # "economy" (default) uses each role's own model; "quality" applies each role's escalation step.
    # Set here, by FACTORY_TIER, or by load_config(tier=...); the effective tier is logged with every run.
    tier: Literal["economy", "quality"] = "economy"
    # Per-role exceptions to the tier, e.g. {reviewer: quality}.
    role_tiers: dict[str, Literal["economy", "quality"]] = {}
    agent_caps: AgentCaps = Field(default_factory=AgentCaps)
    # How plain-code agents are called: "direct" in-process, or "mcp" through the factory's MCP server
    # (factory/mcp), a separate process started without keys in its environment.
    tool_transport: Literal["direct", "mcp"] = "direct"
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)

    root: Path = Path(".")  # project root, set at load time
    models: dict[str, ModelInfo] = {}  # loaded from models_file
    # Each role's escalation step resolved to a full RoleConfig, filled at load time.
    escalations: dict[str, RoleConfig] = {}

    @model_validator(mode="after")
    def _roles_use_known_providers(self):
        for name, role in self.roles.items():
            if role.provider not in self.providers:
                raise ValueError(f"role {name!r} uses unknown provider {role.provider!r}")
        return self

    @property
    def work_dir(self) -> Path:
        return self.root / self.workspace / self.name

    def model_info(self, model: str) -> ModelInfo:
        if model not in self.models:
            raise KeyError(f"model {model!r} is not in the model catalog ({self.models_file})")
        return self.models[model]

    def family(self, role: str) -> str | None:
        r = self.roles.get(role)
        return self.model_info(r.model).family if r else None

    def families(self, role: str) -> set[str]:
        """Every family a role can use: its own model, its fallbacks and its escalation step."""
        steps = [self.roles[role], self.escalations.get(role)]
        return {self.model_info(m).family for r in steps if r for m in [r.model, *r.fallbacks]}

    def separation_errors(self) -> list[str]:
        """Violations of the family-separation rule; empty means the roles are valid. Both tiers count, since a
        run may escalate an author and a reviewer independently."""
        authors = [r for r, rc in self.roles.items() if rc.kind == "author"]
        errors = []
        for reviewer in (r for r, rc in self.roles.items() if rc.kind == "reviewer"):
            for author in authors:
                shared = self.families(reviewer) & self.families(author)
                if shared:
                    errors.append(f"reviewer {reviewer!r} shares {sorted(shared)} with author {author!r}")
        return errors

    def section_hash(self, section: str) -> str:
        """Stable hash of one config section, recorded in run manifests."""
        payload = json.dumps(getattr(self, section).model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def load_models(path: Path) -> dict[str, ModelInfo]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {name: ModelInfo(**info) for name, info in data.get("models", {}).items()}


SPEND_KEYS = ("FACTORY_TIER", "FACTORY_ROLE_TIERS")  # choose a paid tier: the real environment only


def load_dotenv(path: Path) -> dict[str, str]:
    """Set variables from a ``.env`` file (KEY=value lines) that the environment does not already set. Returns every
    variable the file declares, so a caller can check a flag against the file even when the environment shadows it."""
    found: dict[str, str] = {}
    if not path.exists():
        return found
    for line in path.read_text(encoding="utf-8-sig").splitlines():  # utf-8-sig: Notepad's BOM must not join a key
        line = line.strip()
        if line.startswith("export") and line[6:7].isspace():  # `export KEY=value`, as shells write it
            line = line[6:].strip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if sep and key and not key.startswith("#"):
            value = value.strip()
            if value[:1] in ("'", '"') and (end := value.find(value[0], 1)) > 0 and (
                    not (rest := value[end + 1:].strip()) or rest.startswith("#")):  # quoted, then nothing or a comment
                value = value[1:end]
            elif key != "ASPIRE_DATA_PRIVATE":
                value = value.strip("\"'")  # as before for every other key
            # the lab-data flag is kept whole: `0  # off` and a malformed `"0` are not "0", so it reads them as on
            found.setdefault(key, value)  # first occurrence wins here and in os.environ, so the two always agree
            if key == "ASPIRE_DATA_PRIVATE" and _is_private(value):
                found[key] = value  # except the lab-data flag: any line that turns it on counts, whatever the order
                if not _is_private(os.environ.get(key)):
                    os.environ[key] = value
            if key not in SPEND_KEYS:  # never exported, or a later load_config in this process would read them
                os.environ.setdefault(key, value)
    return found


def _is_private(value: str | None) -> bool:
    """The lab-data flag is on for any value but empty or "0", so a typo such as "yes" fails closed."""
    return (value or "").strip() not in ("", "0")


def load_config(path: str | Path, tier: str | None = None) -> FactoryConfig:
    """Load a factory config. ``tier`` (or FACTORY_TIER) overrides the file's tier for this run. Keys are read
    from the environment, or from a ``.env`` file at the project root."""
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    # configs/ sits at the project root; resolve relative paths from there.
    root = path.parent.parent if path.parent.name == "configs" else path.parent
    # The spend settings come from the real environment only, as before: a stale .env line must not move a run to a
    # paid tier. Read them before .env is loaded.
    env_tier, role_tiers = os.environ.get("FACTORY_TIER"), os.environ.get("FACTORY_ROLE_TIERS", "")
    # Load .env before the lab-data guard is read: ASPIRE_DATA_PRIVATE may be set only there, and a flag read
    # before this line would silently stay off.
    dotenv = load_dotenv(root / ".env")
    if tier := tier or env_tier:
        data["tier"] = tier
    # Lab data: Claude and OpenAI refuse it. Either source turning the flag on is enough, so a stale or empty shell
    # value cannot shadow ASPIRE_DATA_PRIVATE=1 in .env.
    if _is_private(os.environ.get("ASPIRE_DATA_PRIVATE")) or _is_private(dotenv.get("ASPIRE_DATA_PRIVATE")):
        data["data_public"] = False
        if not _is_private(os.environ.get("ASPIRE_DATA_PRIVATE")):  # keep os.environ authoritative for later readers
            os.environ["ASPIRE_DATA_PRIVATE"] = "1"
    for pair in filter(None, role_tiers.split(",")):  # "reviewer=quality,coder=quality"
        role, _, t = pair.partition("=")
        data.setdefault("role_tiers", {})[role.strip()] = t.strip()
    unknown = set(data.get("role_tiers", {})) - set(data.get("roles", {}))
    if unknown:
        raise ValueError(f"role_tiers names unknown roles: {sorted(unknown)}")
    cfg = FactoryConfig(**data, root=root)
    for p in cfg.providers.values():  # relative key paths are relative to the project root
        if p.api_key_file and not p.api_key_file.expanduser().is_absolute():
            p.api_key_file = root / p.api_key_file
    models_path = cfg.models_file if cfg.models_file.is_absolute() else root / cfg.models_file
    cfg.models = load_models(models_path)
    unknown_credit = set(cfg.credits.by_provider) - set(cfg.providers)
    if unknown_credit:
        raise ValueError(f"credits name unknown providers: {sorted(unknown_credit)}")
    for name, role in list(cfg.roles.items()):
        cfg.model_info(role.model)  # fail early on a model missing from the catalog
        for fallback in role.fallbacks:
            if cfg.model_info(fallback).real:
                raise ValueError(f"role {name!r}: fallback {fallback!r} is a paid model; fallbacks must be free")
        if role.on_truncation and role.on_truncation not in role.fallbacks:
            raise ValueError(f"role {name!r}: on_truncation {role.on_truncation!r} must be one of its fallbacks, "
                             "so it is free and counts toward family separation")
        if role.quality:
            step = role.model_copy(update={"fallbacks": [], "on_truncation": None, **role.quality, "quality": None})
            if step.provider not in cfg.providers:
                raise ValueError(f"role {name!r} escalates to unknown provider {step.provider!r}")
            if not cfg.model_info(role.model).real and not cfg.model_info(step.model).real:
                # Escalation trades money for quality; between two free models there is no money to save.
                raise ValueError(f"role {name!r}: both tiers are free models ({role.model}, {step.model}); "
                                 "give the role its best free model and no escalation step")
            cfg.escalations[name] = step
            if cfg.role_tiers.get(name, cfg.tier) == "quality":
                cfg.roles[name] = step.model_copy(update={"quality": role.quality})
    if errors := cfg.separation_errors():
        raise ValueError("family separation: " + "; ".join(errors))
    cfg.sandbox.export()
    return cfg
