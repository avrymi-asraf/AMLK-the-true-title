# Wiki LLM Templates

Read this reference only when initializing the wiki or creating a new source or synthesis page.

## `wiki/index.md`

```md
# Wiki Index

## Scope
[What this wiki accumulates and what is out of scope.]

## Topic Map
<!-- Group pages by useful topics. One line per page: what question it helps answer. -->

## Open Tensions and Contested Claims
<!-- Disagreements, unresolved hypotheses, claims needing verification. -->

## Knowledge Gaps and Next Investigations
<!-- Missing sources or high-value unanswered questions. -->

## Recent Material Updates
<!-- Date, source/page affected, and conceptual consequence. Keep compact. -->
```

A useful update item describes a conceptual change:

```md
- 2026-06-03 — Ingested [[source-slug]]; revised [[page-a]] to distinguish X from Y and added a disputed relationship with [[page-b]].
```

## Source Page: `wiki/sources/<source-slug>.md`

```md
---
title: "<Source title>"
type: source
source_kind: paper | article | book | transcript | note | codebase | experiment | web | other
author_or_origin: "<author, organization, user, repository, or unknown>"
published: YYYY-MM-DD | unknown
captured: YYYY-MM-DD
url_or_location: "<URL, path, identifier, or user-provided>"
reliability: primary | secondary | informal | user-provided | unknown
status: processed | partial | needs-review
tags: [<tag>]
---

# <Source title>

## Scope and Relevance
What this source addresses and what it does not establish.

## Faithful Summary
Compact source-bound account. Do not blend in outside knowledge.

## Extracted Knowledge
- **Definition/Object:** <term> — <meaning in source>.
- **Claim:** <atomic claim>.
  - Support: <result, observation, argument, or none stated>.
  - Scope/conditions: <limitations and assumptions>.
  - Status: observed | argued | speculative | unclear.
- **Mechanism/Process:** <how or why account, if present>.

## Limitations and Failure Modes
- <qualification or reason not to overgeneralize>.

## Integration Candidates
- Update [[<page>]] because <specific consequence>.

## Tensions or Contradictions
- <conflict with existing knowledge, or none identified>.
```

## Synthesis Page: `wiki/pages/<page-slug>.md`

```md
---
title: "<Page title>"
type: concept | entity | method | comparison | synthesis | question
status: stub | developing | stable | contested
created: YYYY-MM-DD
updated: YYYY-MM-DD
aliases: [<alias>]
tags: [<tag>]
source_pages: [<source-slug>]
related_pages: [<page-slug>]
---

# <Page title>

## Core Model
What this is and the main idea needed to reason about it.

## Why It Matters
What it helps explain or decide.

## Explanation
Integrated account emphasizing mechanisms, distinctions, conditions, and implications.

## Claims and Evidence
- **Claim:** <synthesized claim>. Evidence: [[<source>]]. Status: established | supported | provisional | disputed.

## Relationships
- **<relation> →** [[<page>]]: <why this relation matters>. Evidence: [[<source>]].

## Boundaries and Failure Modes
What must not be inferred, where the idea fails, or what remains conditional.

## Open Questions or Tensions
- <unresolved issue or missing evidence>.

## Sources
- [[<source-slug>]] — <contribution to this page>.
```

Adapt headings to type: comparisons may use `Dimensions of Comparison`; questions may use `Competing Answers` and `Evidence Needed`; method pages may use `Procedure`, `Preconditions`, and `Failure Modes`.
