# AMLK paper — writing handoff

**Last updated:** 2026-07-29  
**Status:** Abstract and Introduction both revised. Abstract (lines 43–59): narrative pivot framing,
"multi-stage" instead of a specific stage count (fixed a real off-by-one: text said "eight-stage" but the
enumerated pipeline in §3.2/Table 1 has 7 items). Introduction (§1, lines 69–158): rewritten to be more
to the point per the assignment rubric's 4 required questions (no-jargon objective, current practice +
limits, what's new + why it'll work, who cares) — each already had its own `\paragraph{}` header, so this
pass tightened prose and cut jargon rather than restructuring; also swapped one hardcoded citation
("Nallapati et al. (2016) and See et al. (2017)") for `\citet{}` so it can't drift from `bib.bib`, giving
the section both the bracket and non-bracket citation styles the rubric asked for. All 13 figures the
paper cites are now real, generated PNGs in `paper/figures/` — see "Figures" section below. Bibliography
(`bib.bib`) reconstructed and working — no more `?` for unresolved citations. Next: Data / Methods
sections (§2–3), or whichever section you flag next.

## Paper files

| File | Role |
|------|------|
| `paper/main.tex` | ACL draft (restored Jul 29) |
| `paper/main.pdf` | Compiled PDF (stale — regenerate after any `.tex`/figure change; see Building note) |
| `paper/figures/` | All 13 cited figures — real PNGs, built 2026-07-29 (was placeholder/missing before) |
| `paper/bib.bib` | Bibliography — reconstructed 2026-07-29 from the 7 entries already resolved in `main.bbl` (all 7 `\citep`/`\citet` keys in `main.tex` now compile clean, no more `?` in output) |

## Figures — how they're built (2026-07-29)

`paper/` itself is **untracked in git** (not committed to any branch) — that's why figures looked
"missing" earlier in this session; they were never on disk anywhere in this clone, not lost to a
different branch. The obsidian notes (`docs/obsidian/Experiment Results.md`) referenced
figure-generation scripts that didn't exist in this clone (`rubric_figures.py`, `repair_figures.py`,
`baseline_reliability_figures.py`, `supplementary_figures.py`) — someone ran them and got results
into the shared obsidian vault, but the code + images themselves were never committed. Rewrote all
of them in `data_curation/analysis/` this session, validated bit-for-bit (to rounding) against the
numbers already in `main.tex` Table 1 and the obsidian notes. Regenerate with:

```bash
source .venv/bin/activate
python -m data_curation.analysis.figures                       # F1, F2
python -m data_curation.analysis.rubric_results                 # outputs/results/e1_summary.json
python -m data_curation.analysis.rubric_figures                 # F3, F4, F5
python -m data_curation.analysis.repair_figures                 # F6 (dumbbell + heatmap), F7
python -m data_curation.analysis.baseline_reliability_figures   # F9
python -m data_curation.analysis.supplementary_figures          # sx01, sx05, sx16
cp outputs/figures/{f1_curation_funnel,f2_defect_prevalence,f3_rubric_distributions,f4_effect_sizes,f5_length_lead_bias,f6_paired_repair,f6_transition_heatmap,f7_win_rate,f9_baseline_vs_reference_rubric,sx01_filter_overlap,sx05_headline_edit_subtypes,sx16_pilot_kappa}.png paper/figures/
```

F8 (E4 training-comparison figure) stays a placeholder — genuinely blocked on the external
fine-tuning run, not a missing-script issue like the others were.

## Interim fine-tuned-vs-zero-shot finding (2026-07-29, not F8)

Found a 13th artifact while chasing down "different graphs in another branch": confirmed via
exhaustive git archaeology (all branches, `git ls-remote` on the GitHub remote, the stash, full
history search for image files) that **no branch anywhere has additional figures** — this really
was never on a branch. What's actually on disk is `outputs/results/finetuned-by-edit-type.json`
(created 2026-07-29 19:02, ~1h before this session — generating script no longer exists anywhere,
not committed, not stashed): `n=580` matched frozen-test-split rows comparing the **partially
fine-tuned Arm B checkpoint against zero-shot base** (dicta-il/dictalm2.0-instruct) — **not** Arm A
vs. Arm B, since Arm A still doesn't exist. Added as `outputs/figures/sx17_finetuned_vs_zeroshot.png`
/ `paper/figures/sx17_finetuned_vs_zeroshot.png` (script:
`data_curation/analysis/finetuned_baseline_figures.py`), wired into a new appendix section
"Interim fine-tuned vs. zero-shot comparison" (`app:finetuned-interim`), explicitly labeled as not
resolving E4. The finding itself: fine-tuned scores *lower* than zero-shot base on judge
faithfulness/fluency across every edit sub-type and both strata — a caution sign, possibly an
under-trained checkpoint (94% of one epoch, interrupted). Also corrected the E4 results text's
stale "50 of 586 test-set predictions" to "580 of 586" to match this artifact.

## Title & authors

- **Title:** Auditing HeSum: Reference Quality Defects in a Hebrew News Summarization Corpus
- **Authors:** Amit Benbenishti, Avraham Asraf, Ofek Varona

## Project arc (for narrative)

1. **Original plan** — Fine-tune Qwen3-2B on HeSum, probe lead bias (`docs/ANLP Project abstract.md`).
2. **Qwen era** — Model learned HeSum *style* not faithfulness; ROUGE misleading when decode artifacts inflated scores.
3. **DictaLM era** — Hebrew-native base (`dicta-il/dictalm2.0-instruct`) improved fluency; zero-shot outputs often *better* than references.
4. **Pivot** — HeSum targets are scraped extended subheadings, not gold summaries. Paper is a **dataset audit + repair**, not a model-building exercise.

Deep narrative: `docs/obsidian/Project Pivot.md`

## What is done (cite these numbers)

Source: `docs/obsidian/Experiment Results.md`

- **Curation:** 10,000 raw → 5,854 usable (41.5% net drop); 52.4% of survivors needed headline rewrite.
- **E1:** Defect strata worse on rubric; largest effect S2 single-focus Cliff's δ ≈ −0.95.
- **E2/E3:** Curated preferred 73.6% on rewrites; 99.5% tie placebo on untouched rows.
- **F5:** Length/lead overlap is continuous — no discontinuity at 4k token filter.
- **E4:** Training comparison (curated vs original headlines, same articles) — **in progress** at submission time.
- **F9:** Partial DictaLM2 zero-shot baseline (n≈604); S2 single-focus reversal (model more single-focus than defective reference).

## Key repo docs for writing

- `docs/obsidian/Experiment Results.md` — article-ready numbers
- `docs/obsidian/Paper Figures.md` — F1–F9 figure claims
- `docs/obsidian/Data Curation Pipeline.md` — pipeline counts
- `docs/obsidian/Reference Quality Rubric.md` — four judge dimensions
- `docs/superpowers/specs/2026-07-26-dataset-review-experimental-design.md` — pre-registered design

## Current abstract (plain text)

Hebrew news summarization is commonly trained on HeSum, a corpus of article–headline pairs scraped from Hebrew news sites and used as-is for the training target. We argue that a large share of these targets are unfit to train on, and that this, not model capacity, is the field's real bottleneck. The evidence came from two fine-tuning attempts, neither originally designed as a data audit. Qwen3-2B, fine-tuned on HeSum, learned the corpus's headline style without learning to be faithful to the article it summarized. Switching to the Hebrew-native dicta-il/dictalm2.0-instruct fixed fluency, but surfaced a harder problem: its zero-shot summaries routinely read as better than the references they were scored against. Both failures pointed past the model and into the data, so we stopped fine-tuning and audited the corpus instead. We built a multi-stage curation pipeline and a four-dimension rubric judge (faithfulness, single-focus, informativeness, cleanliness) that reads the source article directly and scores a headline without needing a second reference string to compare it against. A large share of the corpus fails this audit outright, and most of what survives still needs its headline rewritten. Flagged rows score measurably worse on every rubric dimension, and in a blind head-to-head comparison curated headlines are strongly preferred over the originals—while a placebo comparison on untouched rows confirms the judge isn't just guessing. We report this as a dataset review rather than a model-building exercise, and close with the training comparison (curated vs. original headlines, identical articles) that was still in progress at submission time.

## Decisions made this round (abstract)

- **Numbers vs. narrative (2026-07-29, round 2):** stripped every specific statistic (10,000 rows,
  5,854/58.5%, 52.4%, Cliff's δ ≈ −0.95, 73.6%, 99.5%) in favor of pure narrative language ("a large
  share", "most of what survives", "strongly preferred", "confirms the judge isn't just guessing").
  All of these numbers still live in the body (Table 1, §Results) — the abstract now only asserts the
  shape of the finding, not its magnitude. If a headline number is wanted back later, the win-rate
  (73.6%) is the natural single candidate — it's the "so what" payoff stat.
- **Tone/length:** rewritten narrative-style (the two-pivot story told in order), ~260 words —
  length treated as secondary to a clear pivot narrative.
- **Stage count:** "eight-stage" was wrong (§3.2's enumerated pipeline has 7 items, matching the
  `stage-6` = headline-curation cross-ref in Table 1). Fixed by going generic — "multi-stage" — in
  the abstract, Introduction, Contributions, and §3.2 opening line, and dropped the numbered
  `stage-6` cross-reference in Table 1 in favor of a name (`headline-curation decision`). If an exact
  count is wanted later, recount the real pipeline stages first (does "Download" or "Final assembly"
  count as a stage?) before reintroducing a number anywhere.
- **E4 framing:** kept as-is — one closing sentence flagging the training comparison as in-progress.
- **F5 (lead-bias/length):** not mentioned in the abstract — stays a body-only result.

## Current Introduction (plain text, §1)

**What we are trying to do.** We check whether the headlines used to train Hebrew news summarizers are actually good enough to learn from, and whether fixing the bad ones changes what a model learns. The dataset in question, HeSum, is one of the few Hebrew resources large enough to fine-tune a modern language model on, so if its targets are flawed, that flaw reaches further than one project.

**How this is done today, and its limits.** The usual recipe for a new-language summarization paper is to fine-tune an instruction model on the available corpus, then report ROUGE, BERTScore, and an LLM judge scored against the corpus's own references — every step of which treats those references as correct. That assumption does not hold automatically for a corpus scraped from news sites: HeSum's own documentation describes its summaries as extended subheadings lifted from articles, not summaries written for the task, and Nallapati et al. (2016) and See et al. (2017) already showed that references built this way tend to be lead-aligned, rewarding a model for copying the opening sentence rather than actually summarizing.

**What is new here, and why we expect it to work.** Instead of running one more fine-tuning attempt on an unaudited corpus, we flip the order: audit the references first, train second. We built a multi-stage pipeline that removes or repairs the worst headlines, and a rubric judge that reads the source article directly and scores a headline on four independent dimensions, rather than comparing it to another string. Because the judge only needs the article, it can score a human-written reference and a model's output on the identical scale, which is what lets us connect the dataset audit directly to a training comparison. We pre-register the four experiments this enables so the paper's conclusions rest on effect sizes decided in advance, not on hindsight.

**Who this matters for.** Anyone training a Hebrew summarizer on HeSum inherits whatever is wrong with it, whether they audit it or not. If the problems are as common and as damaging as we find, the fix is not a bigger model — it is cleaning the targets before training starts. That is a claim about a public dataset that any future HeSum user can act on, not just about the one model we happened to train.

(followed by unchanged §1.1 pivot narrative, §1.2 research questions, §1.3 contributions)

## Decisions made this round (Introduction)

- **Assignment rubric compliance (2026-07-29):** the four required questions (objective in
  plain language / current practice + limits / what's new + why it'll work / who cares) already had
  their own `\paragraph{}` header each — kept that structure since it makes rubric compliance obvious
  to a grader, but tightened every paragraph for concision ("more to the point," per Amit) and cut
  jargon from the first paragraph specifically (dropped "morphologically rich," "training signal") per
  the rubric's "absolutely no jargon" instruction for that one.
- **Citation style variety:** swapped a hardcoded "Nallapati et al. (2016) and See et al. (2017)" for
  `\citet{}` calls (auto-generated, can't drift from `bib.bib`) — gives the Introduction both the
  bracket style (`\citep`, e.g. HeSum, BERTScore) and the non-bracket/prose style the rubric's citation
  guidance asked for.
- **Subsections 1.1–1.3 (pivot narrative, RQs, contributions):** left substantively unchanged — they
  already read as tight, concrete prose rather than rubric filler, so the "more to the point" pass
  focused on the four opening paragraphs.

## Open items for abstract

- [ ] Fix author emails in `main.tex` (still placeholder `<email>`)
- [x] Re-read the new narrative abstract vs. the Introduction's pivot subsection (§1.1) — some overlap
      is expected and fine (abstract compresses, §1.1 elaborates with the three named eras and specific
      numbers 2.98/2.64); no rewording needed.

## Amit's remarks (paste below)

<!-- Add your abstract feedback here -->
