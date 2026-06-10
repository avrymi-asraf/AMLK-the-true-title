# Wiki LLM Quality Gate and Audit Checklist

Read this reference before completing substantial ingestion/refactoring, or during an audit/lint pass.

## Page Quality Gate

Every new or materially revised synthesis page must pass all seven tests:

| Criterion | Pass condition |
| --- | --- |
| Purpose | Represents a recurring question or reusable object of understanding. |
| Boundary | Neither a source dump nor an arbitrary fragment of a broader page. |
| Synthesis | Explains, distinguishes, or integrates rather than listing readings. |
| Grounding | Load-bearing claims are linked to appropriately scoped source pages. |
| Connections | Important relationships state meaning and supporting evidence. |
| Conflict hygiene | Disagreement, uncertainty, and unknowns remain visible. |
| Retrievability | Title, aliases, and index entry let a future agent find the page. |

Revise pages that fail. Do not merely record that they failed.

## Structural Audit: Fix Directly

- Missing `wiki/index.md`, `wiki/pages/`, or `wiki/sources/`.
- Broken `[[page]]` or `[[source]]` links.
- Important synthesis pages omitted from the index.
- Duplicate pages for one concept under aliases or near-synonyms.
- Missing/contradictory metadata that can be corrected from the file itself.
- Orphan pages with no meaningful relation to any indexed concept.

## Semantic Audit: Investigate Before Fixing

- A concept page is only a summary of one source.
- A material claim is unsupported or linked to a source that does not actually support it.
- A claim was generalized beyond its original conditions, metric, population, timeframe, or assumptions.
- A later source challenges a page but the page still presents the earlier claim as settled.
- Conflicting evidence is hidden rather than modeled.
- A page is too broad to retrieve as one concept or too fragmented to explain coherently.
- Links are decorative and do not express a usable relation.
- A central concept appears repeatedly but lacks a durable page.
- The index description no longer tells the truth about a page's role.

## Audit Output

Report:

- corrected structural faults;
- semantic revisions supported by evidence;
- unresolved contradictions or insufficient support;
- pages that should be merged or split;
- the highest-value missing source or next investigation.

Update `wiki/index.md` when an unresolved issue or gap materially affects future work.
