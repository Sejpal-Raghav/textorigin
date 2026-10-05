# Humanizer MVP implementation status

This ledger applies the approved `SPEC.md`, `BACKLOG.md`, and `IMPLEMENTATION-HANDOFF.md` dated 6 October 2026. It records implemented evidence and keeps incomplete work explicit.

| Story/slice | Status | Evidence | Commit |
|---|---|---|---|
| Guarded local rewrite and CLI foundation | Done | Six style/strength combinations, protected-span and overwrite tests | `1e290cf` |
| DeepSeek optional polish and BYOK backend | Done | Adapter failure/credential tests and API contract tests | `fdfdb45` |
| Web humanizer controls and BYOK | Done | Next.js production build | `93440c8` |
| Long rewrite chunking | Done for rewrite; classification aggregation remains | Lossless 100k+ character round-trip and per-chunk service tests | `ef23770` |
| Training preparation | Done; training deliberately deferred | Dataset audit test and memory-aware trainer changes | `1b196f3` |
| Wheel/clean install | Done for local rewrite | Built `textorigin_humanizer-0.2.0-py3-none-any.whl`; installed without source-tree access; command produced valid JSON | pending docs commit |
| GitHub push | Blocked | Local GitHub credential is invalid; commits remain local until authentication is restored | — |

Current focused verification: 35 tests pass. The frontend production build completed and emitted `.next/BUILD_ID`. One upstream Starlette deprecation warning is present in API tests. The full legacy suite currently stops during collection because tomorrow's training environment dependencies (`numpy` and `pandas`, followed by the pinned model stack) are deliberately not installed in this lightweight application environment.

The trained classifier artifact does not exist yet. `classify` and `compare` therefore remain intentionally unavailable in a real demo. No mock classifier output may be presented as real.

Remaining application work before the full specification is complete:

1. Add full-window classification aggregation and actual compare output after the model exists.
2. Add asynchronous job progress and cancellation for the web UI.
3. Route CLI DeepSeek polish through the same similarity-validated service and add masked interactive key entry.
4. Run the full original test suite with all runtime dependencies installed, then exercise a live DeepSeek call with a runtime key.
5. Restore GitHub authentication and push every local commit in order.

Training remains scheduled for 7 October 2026. Run the dataset audit first, inspect cross-family duplicate exclusions, execute a small memory-measured smoke run, and only then start the full three-epoch run.
