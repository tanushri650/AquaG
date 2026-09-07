---
name: Post-merge Python setup
description: Environment behavior needed for Python dependency setup after task merges.
---

Post-merge Python setup must install requirements through the Replit-managed project interpreter with `--break-system-packages`; the default pip invocation is blocked by PEP 668 before the training command can run.

**Why:** The project interpreter uses `.pythonlibs`, while the default environment is marked externally managed and rejects a normal pip install.

**How to apply:** Keep the post-merge script non-interactive, use the project `python`, pass `--no-input` and `--break-system-packages`, then run the project's validation/training command.