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
    token_param: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
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


class Budget(BaseModel):
    """Caps for one run. Dollars are REAL spend only: free ARC calls count as zero, so a free run is never stopped
    by a cost that does not exist. Shadow cost is reported, never enforced."""

    max_real_usd: float | None = None  # hard stop for the run
    # Soft limit: a paid call that would take the run past this needs a person's approval (factory/llm/gate.py).
    approve_above_usd: float | None = None
    max_tokens: int | None = None
    max_seconds: float | None = None


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
        """Every family a role can use: its own model and its escalation step."""
        steps = [self.roles[role], self.escalations.get(role)]
        return {self.model_info(r.model).family for r in steps if r}

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
    data = yaml.safe_load(path.read_text()) or {}
    return {name: ModelInfo(**info) for name, info in data.get("models", {}).items()}


def load_dotenv(path: Path) -> None:
    """Set variables from a ``.env`` file (KEY=value lines) that the environment does not already set."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def load_config(path: str | Path, tier: str | None = None) -> FactoryConfig:
    """Load a factory config. ``tier`` (or FACTORY_TIER) overrides the file's tier for this run. Keys are read
    from the environment, or from a ``.env`` file at the project root."""
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text())
    if tier := tier or os.environ.get("FACTORY_TIER"):
        data["tier"] = tier
    for pair in filter(None, os.environ.get("FACTORY_ROLE_TIERS", "").split(",")):  # "reviewer=quality,coder=quality"
        role, _, t = pair.partition("=")
        data.setdefault("role_tiers", {})[role.strip()] = t.strip()
    unknown = set(data.get("role_tiers", {})) - set(data.get("roles", {}))
    if unknown:
        raise ValueError(f"role_tiers names unknown roles: {sorted(unknown)}")
    # configs/ sits at the project root; resolve relative paths from there.
    root = path.parent.parent if path.parent.name == "configs" else path.parent
    load_dotenv(root / ".env")
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
        if role.quality:
            step = role.model_copy(update={**role.quality, "quality": None})
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
    return cfg
