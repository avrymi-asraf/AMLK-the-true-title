---
name: testing
description: Testing philosophy for this project — behavior tests only, no implementation dictation, heavy tests opt-in.
---

# Testing Philosophy

## The Rule

Automated tests exist to answer one cheap question: **is everything still wired
and does the pipeline run end to end after I changed the code?** They do not
dictate *how* the code works, and they are not where you decide whether the data
is any good.

**The real check is pipeline inspection** (`src/reasoning_pruning/pipeline_inspection.py`
via `scripts/pipeline_inspection.py` or the notebook). That is where you look
at actual G output and ask the questions that matter: does G's reasoning make sense?
Is the unit split clean? Is D removing real filler and picking a real next step?
A unit test cannot judge any of that — do not try to make it. See the [reasoning-pruning]
skill for when it is required and how to run it.

> Keep the automated suite very small. A few comprehensive "it runs / it's
> connected" tests beat a pile of tiny tests that pin down implementation
> details nobody cares about. If a test would break on a harmless refactor that
> keeps the same behavior, it is the wrong test — delete it.

## What to test (a few comprehensive tests, total — not per feature)

The behavioral contracts, at the level the caller cares about:

- Does the pipeline produce `input_x -> target_y` rows and convert to training format?
- Does the next-context invariant hold (`next_context == input_x + "\n" + target_y`)?
- Does the pipeline stop gracefully / return empty when nothing is prunable?
- Are bad attempts discarded without polluting the retry context?
- Does spectrum question assembly never leak answer fields to G?
- Does a stubbed/live LLM response parse into a usable decision?

## The pipeline inspector must run the production loop — never a copy

The pipeline inspector must execute `build_rows_for_question` (via the
`PruningObserver` hook), not a hand-copied parallel loop. We learned this the
hard way: a copied inspection loop silently drifted (production required ≥3 units
while the inspection copy still allowed ≥2), so the "most important verification"
was no longer running what production ran. One loop, an observer for printing —
that is the only acceptable shape. This matches the project's no-parallel-paths
rule ([feedback_no_backward_compat]).

## What NOT to test

**Implementation details that don't matter to the caller:**

- Exact field names in `metadata` (the caller only needs `input_x`/`target_y`)
- Internal context string formats (`context_before_generation` exact value)
- Unit-splitting internals (which strategy produced which list)
- CLI parser argument structure
- Script source code contents (`assert "load_dataset" in script.read_text()` — never do this)
- Config YAML field values (those are config, not behavior)
- Exact URL formats in HTTP calls
- Internal attribute names on model clients

## Heavy tests (model loading) — always opt-in

Any test that loads a transformer model or calls a live API **must** be guarded:

```python
@pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") and os.getenv("RUN_LIVE_TESTS")),
    reason="Set GEMINI_API_KEY and RUN_LIVE_TESTS=1 to run live LLM tests",
)
def test_live_gemini_pipeline(): ...
```

Default `uv run pytest` must complete in seconds on any machine, including those without a GPU or API key. If it crashes or hangs, the test suite has a guard missing.

## Fake models for fast tests

Shared fakes live in `tests/fakes.py` (`FakeGenerator`, `FakeDecisionModel`,
`make_config`) and satisfy the generator/decision-model protocols. Import them
flat (`from fakes import ...`) — pytest's prepend import mode puts `tests/` on
`sys.path`, so no `__init__.py` is needed. The fake trace has at least 3 units so
a removal at the front leaves a valid target.

## Test files — split by area, so a change in one part runs its own file

- `tests/test_data_creation.py` — the core pruning loop (valid rows, next-context
  invariant, empty-when-nothing-prunable, discard-and-retry).
- `tests/test_question_source.py` — spectrum question assembly never leaks answers.
- `tests/test_clients.py` — G/D wiring: stubbed Gemini decision parsing + the live
  gated end-to-end test.
- `tests/test_pipeline_inspection.py` — smoke that the inspector runs the
  production loop and returns rows.
- `tests/fakes.py` — shared fakes (not a test file).

Do not pile everything back into one file. If you changed only the clients, you
should be able to run only `tests/test_clients.py`.
