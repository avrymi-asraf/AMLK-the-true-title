# AMLK shared Obsidian vault

This folder is a **shared knowledge base** for the AMLK Hebrew summarization project. Open it as an Obsidian vault so the team can browse linked notes, graph view, and search.

## How to open

1. Install [Obsidian](https://obsidian.md/).
2. **Open folder as vault** → select `docs/obsidian/` in this repo.
3. Start from [[Home]].

## Conventions

- Notes use `[[wikilinks]]` between topics.
- Code paths are relative to the repo root (`AMLK-the-true-title/`).
- Results under `outputs/results/` are gitignored; paths are still cited for local runs.
- Status tags: `#status/done` `#status/planned` `#status/in-progress` `#status/blocker` `#status/superseded`

### `#status/superseded`

Marks a note whose conclusions were overtaken by later work. Superseded notes are **kept, not deleted** —
the project changed direction twice and the reasoning behind each turn is part of the research record.

A superseded note carries a blockquote banner directly under its title stating which era it belongs to,
what still holds, and where current work lives. Read the banner before trusting anything below it. The
Qwen-era notes listed in [[Home]] are all in this state; [[Project Pivot]] explains why.

## Sync

Commit and push this folder like any other project docs. Everyone on the team opens the same `docs/obsidian/` path.
