# Working rules for Claude in this repo

## Review every PR before merge

Every pull request gets a Claude code review before it is merged. Reviews run on Parker's Claude
subscription, never on API credit: do not use `ANTHROPIC_API_KEY`, and never call the paid-model code
paths in `factory/` as part of a review.

1. Before you tell Parker a PR is ready, run the `/code-review` skill on it (effort `medium`; use `high`
   for changes to guards, approval gates, spend limits, the sandbox, or the judge/QA prompts). Fix every
   real finding and push; note any finding you rejected and why in the PR description.
2. The `Claude review` GitHub workflow (`.github/workflows/claude-review.yml`) also reviews each PR
   automatically once the `CLAUDE_CODE_OAUTH_TOKEN` repo secret is set. Read its comments and address them.
3. A review is advisory. Parker approves merges.
4. Checks to look at in particular: spend/approval gates still fire, regulated or lab data never reaches
   Claude/GPT, tests that were skipped or weakened, and new code paths that call a paid provider.
