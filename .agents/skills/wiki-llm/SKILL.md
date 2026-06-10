---
name: wiki-llm
description: Use when building or maintaining a durable research wiki from papers, articles, notes, transcripts, code findings, or web research; when asked to ingest sources, connect concepts, synthesize knowledge, query prior understanding, or audit wiki quality. Maintains only wiki/sources/, wiki/pages/, and wiki/index.md. Prioritizes evidence-grounded synthesis, correct page boundaries, explicit conceptual relationships, and cumulative understanding over summarization.
---

# Wiki LLM — Compile Information into Understanding

## Mission

Build a wiki that becomes a better model of its domain after every source and every serious question.

Do not optimize for collecting summaries. Optimize for evidence-grounded claims, reusable concept pages, explicit relationships, visible uncertainty, and fast future retrieval.

Sources are evidence. Pages are current synthesis. The index is the navigation and maintenance surface.

## Fixed Wiki Structure

Use exactly:

```text
wiki/
├── index.md
├── sources/
│   └── <source-slug>.md
└── pages/
    └── <page-slug>.md
```

Do not create `raw/`, `concepts/`, `entities/`, `queries/`, `log.md`, `SCHEMA.md`, or another wiki layout unless explicitly requested. This skill is the schema; `wiki/index.md` records navigation, gaps, tensions, and important recent updates.

Read `references/templates.md` when initializing the wiki or writing a new source/page. Read `references/quality-gate.md` during audits or before completing any substantial ingest/refactor.

## Knowledge Model

### `wiki/sources/`: evidence layer

One source page per paper, article, note, transcript, dataset/report, experiment result, code investigation, or web source. Record provenance, faithful extraction, scope, limitations, and potential integrations.

A source page can be valuable without justifying a synthesis page.

### `wiki/pages/`: synthesis layer

One page per reusable object of understanding:

- `concept`: definition, principle, mechanism, phenomenon;
- `entity`: system, model, project, dataset, organization, person;
- `method`: process, algorithm, workflow, technique;
- `comparison`: stable trade-off or contrast;
- `synthesis`: explanation spanning multiple concepts;
- `question`: important unresolved or disputed issue.

Pages are about ideas, not documents. A page must integrate and connect; it must not merely echo one source.

### `wiki/index.md`: map and control layer

The index contains scope, topic map, discriminative page descriptions, open tensions, knowledge gaps, and compact material-update history. Read it first and update it after any material change.

## Epistemic Rules

1. Separate evidence from synthesis: source pages state what sources support; pages state the integrated model.
2. Make load-bearing claims traceable to source pages.
3. Preserve claim status where it matters: `established`, `supported`, `provisional`, `disputed`, `inferred`, `unknown`.
4. Preserve disagreement; identify exactly what conflicts and what evidence might resolve it.
5. Do not treat repeated assertion as independent evidence.
6. Preserve applicability boundaries: assumptions, conditions, metrics, dates, population, and failure modes.
7. Use links to encode meaning, not term co-occurrence.
8. Prefer surgical edits; do not overwrite still-valid synthesis.
9. Treat missing evidence as a recorded gap, not a license to guess.

## Page Boundary Rules

Create a page when an idea is independently queryable, foundational to other explanations, supported by multiple meaningful claims/relations/sources, represents an important comparison or mechanism chain, or needs a durable place for unresolved evidence.

Merge into an existing page when new information sharpens the same definition, mechanism, question, entity, trade-off, or conclusion.

Do not create a page for every source, incidental mention, isolated example, temporary answer, or conclusion that naturally belongs inside an existing page.

Split a page when it answers independent questions, is no longer retrievable as one unit, or repeated updates affect only a separable section. Preserve links between the resulting pages and update the index.

## Relationship Protocol

Material links must name the relationship and why it matters. Prefer relations such as:

`is-a`, `part-of`, `requires`, `enables`, `causes`, `mediates`, `measures`, `operationalizes`, `generalizes`, `contrasts-with`, `competes-with`, `contradicts`, `explains`, `applies-to`.

Use this form:

```md
- **Requires →** [[related-page]]: depends on ... Evidence: [[source-slug]].
- **Contrasts with →** [[alternative-page]]: both address ..., but differ in ... Evidence: [[source-a]], [[source-b]].
```

Do not add a link unless following it would help answer a realistic future question.

## Always Orient Before Writing

For every ingest, query, integration, or audit:

1. Confirm the fixed wiki structure exists; initialize only if the task requires a wiki and it is absent.
2. Read `wiki/index.md`.
3. Search existing pages and sources for central terms, aliases, and adjacent concepts.
4. Read relevant pages before deciding whether to create, merge, relate, split, or record a tension.

Never create a page before checking whether the idea already exists under another name or as part of a broader page.

## Initialize

Create only `wiki/index.md`, `wiki/pages/`, and `wiki/sources/`. Use the index template in `references/templates.md`.

## Ingest a Source

Ingestion is a reasoning pipeline, not a summary command.

### Pass 1: Capture source evidence

Create or update `wiki/sources/<source-slug>.md` using the source template. Faithfully record source claims, support, scope, limitations, and integration candidates. Do not blend later inference into source-reported content.

Do not paste long copyrighted text. Capture metadata, faithful paraphrase, necessary short excerpts only, and a reference to the original.

### Pass 2: Extract what can change understanding

Identify the source's central question, definitions, atomic claims and support, mechanisms, assumptions, applicability boundaries, results, contradictions, gaps, and affected existing concepts.

Be selective: retain information that may alter future reasoning or decisions; omit detail that only adds storage.

### Pass 3: Map each knowledge unit

Classify each extracted unit before editing synthesis pages:

- `ignore`: out of scope or not durable;
- `source-only`: worth preserving but not synthesis-worthy yet;
- `merge`: improves an existing page;
- `new-page`: establishes a reusable concept/entity/method/comparison/synthesis/question;
- `relationship`: adds a meaningful connection;
- `tension`: challenges or narrows an existing conclusion;
- `gap`: creates an important investigation item.

### Pass 4: Integrate into pages

For every materially affected page: preserve stable explanation, revise only affected sections, link important claims to sources, add semantic relationships, state conflict or narrowed scope, and refresh metadata.

Create a new page only after applying the boundary rules. Use the page template in `references/templates.md`.

### Pass 5: Update the index

Update topic-map entries, tensions, gaps, and one compact recent-update item describing the change in understanding rather than only listing edited files.

## Query the Wiki

1. Read the index; identify relevant pages, tensions, and gaps.
2. Read the smallest relevant set of synthesis pages.
3. Verify load-bearing claims through their linked source pages.
4. Answer from the wiki's model with page/source citations.
5. State clearly when evidence is absent, narrow, or conflicting.
6. Identify durable new relationships, clarifications, or gaps exposed by the query.

Update the wiki from a query only when the result is reusable synthesis: a recurring explanation, comparison, sharpened/resolved tension, or important new relationship. When the task is only Q&A, answer first and propose the durable update rather than silently expanding the wiki.

## Research to Fill a Gap

When external research is required:

1. Start from a named gap, disputed claim, or question.
2. Seek sources capable of changing the answer: primary records, official documentation, papers, datasets, or strong analyses.
3. For contested topics, deliberately seek credible counter-evidence.
4. Ingest each useful source through the full pipeline.
5. Revise synthesis only after comparing new evidence against the existing model.

Search snippets are leads, not evidence. Read and capture a source before it supports durable wiki claims.

## Audit / Lint

Fix structural faults directly: missing required structure, broken links, important pages absent from the index, clear duplicate pages, clearly inconsistent metadata, and orphan pages with no useful relation to the wiki.

Investigate semantic faults before fixing: source-shaped summaries pretending to be concepts, unsupported or overgeneralized claims, stale conclusions, hidden contradictions, over-broad or over-fragmented pages, meaningless links, missing central pages, and stale index descriptions.

Use `references/quality-gate.md` for the full audit checklist and page pass criteria. Reflect important unresolved findings in the index.

## Conventions

- Use lowercase kebab-case filenames.
- Use `[[page-slug]]` and `[[source-slug]]` links; explain material relationships in prose.
- Name synthesis pages after concepts or questions, not source titles.
- Use ISO dates: `YYYY-MM-DD`.
- Label model-generated conclusions explicitly as `Inference:` or `Hypothesis:` when not directly supported.
- Keep index descriptions concise and discriminative.
- Update only the fixed wiki structure unless explicitly told otherwise.

## Completion Report

After ingest, synthesis update, or audit, report:

```md
## Wiki Update
- Sources captured: <files>
- Pages created: <files or none>
- Pages materially revised: <files or none>
- New or changed relationships: <brief list>
- Tensions or uncertainty preserved: <brief list or none>
- Highest-value next investigation: <one item or none>
```
