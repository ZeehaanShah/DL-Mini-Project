# AGENTS.md — Persistent Agent Instructions

## Mandatory Workflow for Every Agent Session

1. **Read `PROJECT_CONTEXT.md` before starting any task.**
   - Understand current project state, completed work, in-progress items, and blockers.
   - Do not duplicate completed work or contradict documented decisions.

2. **Inspect the actual repository before changing files.**
   - Verify file existence, content, and structure before modifications.
   - Do not assume file contents from memory or prior sessions.

3. **Update `PROJECT_CONTEXT.md` after every meaningful task.**
   - Record: files changed, reason, checks performed, results, next steps.
   - Mark items as COMPLETED, IN-PROGRESS, BLOCKED, or PLANNED.
   - Timestamp significant updates.

4. **Update `CHANGELOG.md` for every change.**
   - Date, description, files affected, rationale, validation performed.

5. **Never report a task as complete unless verified.**
   - Code must be implemented and tested, not just planned.
   - Metrics must come from actual runs, not fabrications.

6. **Preserve existing data and code unless change is documented.**
   - Raw data files must never be modified.
   - Explain why any existing code or config is being changed.

7. **Never fabricate:**
   - Dataset contents, schemas, or row counts.
   - Training metrics or experiment outcomes.
   - Citations or paper references.
   - API responses or external data.

8. **Security:**
   - Never expose API keys, passwords, tokens, or secrets in any file.
   - Use environment variables or `.env` (which must be in `.gitignore`).

9. **Maintain clear work status:**
   - COMPLETED: Implemented, tested, documented.
   - IN-PROGRESS: Partially done, state described.
   - BLOCKED: Cannot proceed, reason documented.
   - PLANNED: Not yet started.

10. **Update PROJECT_CONTEXT.md before ending a session.**
    - Summarize what was done, what remains, and how to resume.

## Data Integrity Rules

- Do not start model training without a completed data audit.
- Do not merge datasets from different sources without verifying location, temporal, and unit compatibility.
- Do not use test data for any model selection or hyperparameter tuning.
- All splits must be documented with exact sample counts and timestamp ranges.
- Leakage checks must be performed and documented before training.

## Code Standards

- All scripts must be runnable from the project root.
- Use configs/ for hyperparameters and experiment settings.
- Use deterministic random seeds; document any nondeterministic operations.
- Keep imports at file top; avoid circular imports.
- Document functions with docstrings.
- Use type hints for function signatures.
