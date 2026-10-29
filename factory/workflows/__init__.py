"""Task types beyond the bug fix, each a template plan and a run over the orchestrator's agents and checks.

- ``test_gen``: write tests for existing code; plain code proves they pass and scores them by the mutants they catch.
- ``pr_review``: review a diff; plain code reads it and runs the tests, a model comments, and only comments on
  changed lines are kept.
"""

from factory.workflows import pr_review, test_gen

TEMPLATES = {"test_gen": test_gen.TEMPLATE, "pr_review": pr_review.TEMPLATE}
WORKFLOWS = {"test_gen": test_gen.run, "pr_review": pr_review.run}
TASK_TYPES = {"bug_fix": "Fix a bug", "test_gen": "Write tests", "pr_review": "Review a pull request"}

__all__ = ["TASK_TYPES", "TEMPLATES", "WORKFLOWS"]
