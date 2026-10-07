## Summary

What does this change do?

## Research Question

Which hypothesis, gate, or finding does this change support?

## Evidence

- Tests:
- Generated results:
- Relevant result files:

## Validation

List the exact commands used to validate the change.

```bash
uv sync --locked
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
uv run pytest -q
```

## Scope and Limitations

What does this change not establish? Note any unresolved validity, reproducibility, licensing, or compute concerns.

## Checklist

- [ ] The change has a focused purpose.
- [ ] Tests exercise the behavior rather than constants.
- [ ] Result files include the full configuration.
- [ ] Negative and failed results are preserved.
- [ ] Documentation and reproduction commands are updated.
- [ ] No claim exceeds the evidence produced by this change.
