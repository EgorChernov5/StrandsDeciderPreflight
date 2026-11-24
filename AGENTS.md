## DTOs, contracts, and constants

Do not take into account this section from the upper-level instructions.

## Build and project commands

Run all commands that require the project's Python environment or dependencies through `uv`.

Allowed format:

```bash
uv run python ...
uv run <command>
```

If the exact command depends on the project configuration, inspect the existing configuration files first and use the command already defined by the project.
---

## Dependencies

The project's package manager is `uv`.
---

## Git Restrictions

- NEVER run `git commit`.
- NEVER run `git push`.
- NEVER create Git commits, including amend or empty commits.
- Do not create commits unless the user explicitly requests it.
- Leave all changes uncommitted for manual review.
---