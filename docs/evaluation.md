# Evaluating ASPIRE

ASPIRE claims three things: it spends the minimum model money per solved task, what it produces reruns with no
model, and nothing it does (a patch, a proposed prompt, a model's reply) can loosen the checks that judge it. Each
claim has an experiment, and each experiment is a command that writes `bench/results/<time>-<name>/summary.md`.

| Claim | Experiment | Needs | Command |
|---|---|---|---|
| Economy | Benchmark, ASPIRE vs one model call, same coder model | ARC | `python -m factory.bench --repeat 3` then `python -m factory.evaluate report bench/results/<time>` |
| Correct, not just green | Hidden tests on every solved patch | nothing | part of the benchmark |
| Reproducible | Each task's bundle reruns under a guard that stops any import of ASPIRE or a model SDK and any socket | nothing | `python -m factory.evaluate rerun` |
| Gated | Hostile patches with every model assumed compromised (coder sends it, reviewer accepts) | nothing | `python -m factory.evaluate guards` |
| Cross-family review | Reviewer models on wrong patches that pass every visible test, and on correct ones written differently | ARC | `python -m factory.evaluate reviewers --authored 2` |
| Full pipeline | Interviewer to deliverable on a public task (NCBI gene to PCR primers) | ARC, NCBI | `python -m factory interview ...`, then `plan`, `research`, `build` |
| Self-extension | A proposed prompt judged on the held-out tasks | ARC | `python -m factory.extend --agent coder` |

## The benchmark's judge

A task is solved when the patch applies, changes no test file, pytest configuration or `.env`, the whole suite
passes, and every test that ran before the patch still runs. The last rule was added after `guards` showed the
judge scoring three hostile patches as solved with nothing run: one that calls `os._exit(0)` at import, and new
`.pytest.ini` and `pytest.toml` files (pytest 9 reads both) that switch collection off. The bug-fix workflow's own
check already caught all three, so only the benchmark's baseline column could have been inflated. Its test-file
rule now knows every file pytest reads its configuration from.

## Hidden tests

`bench/hidden/<task>/` mirrors the task repository and is copied over it after the patch, at judging time only. No
model sees it: the baseline prompt and ASPIRE's coder read the task repository, and self-extension's proposer never
sees held-out tasks at all. The benchmark reports `solved` (visible tests; the verdict self-extension's gate uses)
and `hidden_solved` (strict). `tests/test_evaluate.py` checks that the hidden tests fail on every unpatched task,
pass every correct patch in the review set, and fail every wrong one.

## The review set

`bench/review_set/` holds 35 labeled patches for the ten distinct bug-fix tasks, all of which pass every visible
test, so the fix check cannot tell them apart and only a reviewer (or hidden tests) can:

- 20 wrong: special-cased inputs (`hardcode`), half fixes (`partial`), changes outside the issue (`scope`), changed
  behavior elsewhere (`behavior`), and a value that compares equal to anything (`always_equal`).
- 15 right: the ten golden fixes and five correct alternatives written another way, so a reviewer that rejects
  anything unfamiliar is caught too.

`reviewers --authored N` also has the coder write N patches per task and labels them by the hidden tests, so a
GLM reviewer judging GLM's own work (the same-family case the config forbids) is measured on real output, not only
on planted patches. The experiment builds its own reviewer config per model and skips the family rule on purpose;
the real config is never changed.

## Results (live, 2026-11-02 to 11-03)

Free ARC models on the laptop, $0 real spend; raw notes in the project folder under `evaluation/live/`; the full
write-up is the evaluation report linked from the project.

| Run | ASPIRE (hidden tests too) | Single call (hidden tests too) | Tokens, ASPIRE vs single call |
|---|---|---|---|
| Main tier, 11 tasks x 3 (main 6eebd5d) | 31/33 | 31/33 | 63,393 vs 16,281 (3.9x) |
| Hard tier, 8 tasks x 3 (`bench/tasks_hard.yaml`) | 18/24 | 11/24 (paired 8 to 1, p = 0.04) | 271,405 vs 22,970 (12x) |
| Reviewers, 54 patches each | cross-family: 0/21 wrong accepted; GLM (same family): 1/21 | 0/33 good rejected by any | DeepSeek 94k, gpt-oss 129k, GLM 27k |
| Guards | 9/9 plain-code attacks blocked by the judge and the fix check | 2/2 test-passing attacks caught by hidden tests | - |
| Rerun | 32/32 expected verdicts, 0 model calls | median 2.4 s | $0 |

On one-file bugs the two modes tie and ASPIRE's extra tokens are earned back after about 5 reruns of its bundle; on
harder bugs ASPIRE solves clearly more. Earlier runs are superseded: the first (26/33 each) gave ASPIRE only two free
attempts, and the second (26/33 vs 32/33) came before the coder was shown the modules the failing tests import (#40).

## Reading the numbers

Eleven tasks is a small sample. The cost report gives 95% Wilson intervals: 31 of 33 is 80% to 98%, so the main tier
cannot separate the modes; the hard tier's paired count can. Run with `--repeat 3` for variance across trials.
