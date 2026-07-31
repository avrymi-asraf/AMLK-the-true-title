# AMLK paper — writing handoff

**Last updated:** 2026-07-31
**Status:** Massive three-page rewrite complete, plus a Figure 4/5 fix-up round and a figure-gallery
follow-up that added Figure 3 (E1 rubric score distributions) back in. The assignment was clarified to
require three pages of text while allowing unrestricted figure pages. `main.tex` is now a coherent
audit-first paper: pages 1--3 contain the abstract and complete main narrative; page 4 begins Appendix A
(figures/tables); pages 4--6 contain seven claim-carrying visuals plus Table 1, with Appendix B
(lead-bias check) and the full References list packed onto the last of those pages. All appendix
figures/table use `[H]` (`float` package) rather than `[!htb]` so LaTeX packs them tightly without
reordering text ahead of delayed floats. Qwen history, the chronological "eras," duplicate
RQ/contribution lists, zero-shot side probes, interim Arm-B output, and prose appendices were removed. E4
remains as a compact, replaceable Methods paragraph plus a reserved Results paragraph until both arms
finish. Lead bias is no longer a second body narrative: S1 gets one sentence explaining why length is
treated continuously, and the F5 graph appears alone in Appendix B. The body uses the ACL template's
standard margins; the earlier 1.8 cm margin override was removed because it compressed the rewritten body
to only two pages. Figure 4 (E3 blind preference) was simplified to drop the placebo bar (the placebo
number stays in the prose). Figure 5 was swapped from the judge's self-consistency test-retest plot to a
real (partial) judge-vs-human spot-check (F9a), since the old figure was not what it looked like — see
"Figure 4/5 fix-up round" below. Figure 3 (rubric score distributions by stratum) was reviewed from a full
gallery of 12 candidate figures and added back into Appendix A; the E2 dumbbell plot was reviewed and kept
cut (weak visual, ceiling-compressed). `paper/main.pdf` was regenerated with Tectonic and visually
inspected page by page — now **6 pages** total. No unresolved references or citations remain. Pending:
insert final E4 results once both training arms finish.

## Paper files

| File | Role |
|------|------|
| `paper/main.tex` | Current three-page-text paper source |
| `paper/main.pdf` | Current compiled PDF: 3 text pages + 5 figure/table pages + 1 reference page |
| `paper/figures/` | Figure inventory; six figures are cited in the current compact paper |
| `paper/bib.bib` | Working bibliography; all current citations resolve |

## 2026-07-31 Figure 4/5 fix-up round

- **Figure 4 (E3 blind preference, `f7_win_rate.png`) — placebo bar removed.** Amit asked to "remove the
  placebo section" from the figure. `build_f7_win_rate()` in `data_curation/analysis/repair_figures.py`
  previously plotted two stacked-bar rows (Rewritten, Placebo); now it plots only the Rewritten row. Also
  dropped the redundant "Curated win rate 95% CI" annotation (it was overlapping the subtitle once the
  figure got shorter) since the same CI is already in the caption. The placebo statistic (99.5% tie) is
  unchanged and still lives in the Results prose and the figure caption ("the placebo cohort is reported
  in the text") — only the visual bar was dropped, not the finding.
- **Figure 5 replaced: judge test-retest kappa → real human spot-check (F9a).** Amit asked "is it the
  human check we done? i dont think i got the test" — correctly: the old Figure 5 (`sx16_pilot_kappa.png`)
  was the *judge's own* test-retest reliability (re-scoring the same 300 pilot rows at nonzero
  temperature), not a human check at all. There is a real human validation effort, F9a
  (`data_curation/analysis/human_validation_results.py` / `human_validation_figures.py`,
  `data_curation/artifacts/human_annotations/`, worklist in
  `data_curation/artifacts/human_validation_worklist.json`), but it's disjoint-split across three
  annotators (amit/avreymi/ofek) and **only Amit ever submitted his portion** (122/122 tasks; avreymi
  0/65, ofek 0/60 — confirmed via `python -m data_curation.analysis.human_validation_results --check`).
  Because the split is disjoint, human-human kappa can't be computed at all, and judge-vs-Amit kappa
  numbers (0.18-0.76) plus near-chance pairwise agreement (kappa -0.02) are the only human-anchored
  numbers, but Amit found the existing kappa-based F9a figure/pipeline "overwhelming" and asked for
  something simpler — so instead of reusing `human_validation_figures.py`'s kappa heatmap, wrote a new,
  minimal script **`data_curation/analysis/human_check_figure.py`** that reports plain percentages with no
  kappa: exact-match % and within-1-point % per rubric dimension (from Amit's 52 rubric rows vs. the E1
  judge), plus one plain pairwise agreement % (Amit's 22 pairwise rows that overlap `e3_pairwise.jsonl`
  vs. the E3 judge outcome). Output: `outputs/figures/f9a_human_agreement.png`, copied to
  `paper/figures/`. Wired in as the new Figure 5 (`fig:humancheck`, was `fig:kappa`); caption is explicit
  that this is a single-annotator spot-check, not a full validation study. Added
  `\label{sec:limitations}` to the Limitations section and a new paragraph there stating the honest
  numbers (within-1 mostly high, exact-match under half, pairwise ~45%) and that two of three annotators
  never submitted — trimmed twice to keep the main narrative on 3 pages (final trim also tightened one
  Discussion sentence and the closing Limitations sentence). The Methods paragraph that used to cite
  `Figure~\ref{fig:kappa}` for the judge's test-retest numbers now just states the numbers inline and
  points to `Section~\ref{sec:limitations}` for the human comparison. Deleted the now-unused
  `paper/figures/sx16_pilot_kappa.png` (the underlying `outputs/figures/sx16_pilot_kappa.png` and its
  generator in `supplementary_figures.py` are untouched — still a valid judge-QA artifact, just no longer
  cited in the paper).
- **2026-07-31 follow-up:** removed all annotator-identifying text ("Amit") from the figure subtitle and
  source note (now "One annotator" / "Source: F9a human annotations vs. judge outputs") — the author
  byline in `\author{}` was left alone since that's a normal credit line, not an annotation reference.
  Also removed the "too small to serve as validation" / "Not a full human validation study" framing
  from both the figure caption and §6, per Amit: annotator coverage is being extended (not staying this
  small), so don't frame it as a permanent inadequacy — caption now says "annotator coverage is being
  extended" instead. Restructured §6 into a lead summary paragraph (recaps E1/E2/E3 + the spot-check +
  the auditing-should-precede-training takeaway) followed by one shorter limitations paragraph, instead
  of limitations-heavy prose split with the summary tacked on at the end — merged the old closing
  "Taken together..." sentence into the new opening paragraph to avoid saying the HeSum-defects/curation
  finding twice. Re-verified still exactly 3 text pages after both edits.
- **2026-07-31 second follow-up: removed Discussion section, clarified Methods experiments, filled freed
  space.** Amit noted the assignment guide only lists Abstract/Introduction/Data/Methods/Results (no
  Discussion), and asked that anything important in the standalone `\section{Discussion}` be folded into
  the summary instead. Deleted the Discussion section; its two load-bearing points were merged into
  §5 (now "Limitations and Conclusion", auto-renumbered from §6 after the deletion): the methodological
  point ("reference quality varies systematically... which can reverse how a model score should be
  read... article-grounded evaluation asks whether the target itself is faithful") went into the opening
  summary paragraph, and the S4-editorial-preference nuance ("the curator rewrote many headlines the
  independent judge already scored near the clean group") went into the second (limitations) paragraph
  next to the existing editorial-style sentence, avoiding a duplicate sentence. Also used the page-3 room
  freed by the Discussion cut, per Amit's ask to make Methods clearer about "what are the experiments":
  split the old single `\paragraph{Experiments.}` (which crammed E1+E2+E3 into two paragraphs) into four
  explicit `\paragraph{}`s, each named and phrased as the question it answers -- "E1: Are the flagged
  strata worse than clean headlines?", "E2: Does the paired repair raise a headline's own score?", "E3: Do
  blind readers prefer the repaired headline?", "E4: Does training on the repaired targets change model
  quality?" -- plus one lead-in sentence naming all four before the first paragraph. Also spent a little of
  the freed room on two Data-section rubric-compliance additions: added the `paz-argaman2024hesum` citation
  directly in §2's opening sentence (previously only cited in the Introduction) since the assignment
  explicitly asks the Data section to "reference the original paper," and added the actual train/val/test
  split size for E4 ($4{,}683$/$585$/$586$, $80/10/10$) which was previously only described qualitatively
  ("frozen test split") in Methods. Verified page 3 is now fully used (no trailing whitespace) and the main
  text still ends exactly on page 3, page 4 starts the appendix.
- **2026-07-31 third follow-up: compacted the appendix (fewer near-empty pages).** Amit asked for less
  whitespace in the appendix without shrinking images too much. Root cause: each `[ht]` figure at
  0.82-0.95`\textwidth` was tall enough that only one (sometimes two) fit per page, and there were four
  `\clearpage`s forcing extra breaks on top of that, so Figures 2 and 3 each sat alone on their own
  nearly-blank page. Fix: changed float placement to `[!htb]` (lets LaTeX pack more aggressively) and
  trimmed each image's `\includegraphics` width modestly (0.95→0.78 for the funnel, 0.9→0.72 for E1 effect
  sizes and the F9a spot-check, 0.82→0.62 for the near-square E2 transition heatmap, 0.82→0.72 for the E3
  win-rate bar, 0.92→0.8 for the lead-bias supplementary check) plus a small `\vspace{-0.3em}` before each
  caption -- all still comfortably readable, just not using more width than the content needs. Removed the
  two `\clearpage`s that were forcing Table 1→Figure 2 and Figure 3→Figure 4 onto separate pages (kept the
  `\clearpage` before Appendix B and before the bibliography, since **removing** the one before Appendix B
  caused LaTeX to float the "B Supplementary Lead-Bias Check" heading itself onto an earlier page, ahead of
  its own figure -- a broken-ordering regression, reverted immediately). Result: 6 figures + Table 1 now
  pack into 2 appendix pages (was 5 pages, one page per figure); Appendix B's lead-bias figure keeps its
  own page (nothing else fits alongside it) but the overall paper dropped from 9 pages to **7 pages**.
  Verified the main text still ends exactly on page 3 and every figure/table/heading is still in its
  original left-to-right, top-to-bottom source order (no more float-reordering artifacts).
- **2026-07-31 fourth follow-up: removed the `<email>` placeholders.** Amit asked whether emails were
  needed and whether they'd been made up — confirmed no, `<email>` was always the untouched literal
  ACL-template placeholder, never a fabricated address. Since the assignment rubric doesn't ask for author
  contact info, removed the `\texttt{<email>}` line under each author name in `\author{}` entirely
  (`main.tex`) rather than inventing or leaving placeholder text — now just three names, no more
  `<email>`-shaped loose end. Paper still compiles clean at 7 pages.
- **If Avreymi/Ofek later submit their F9a annotations:** re-run
  `python -m data_curation.analysis.human_check_figure` (it re-reads
  `data_curation/artifacts/human_annotations/amit.jsonl` directly — extend it to also read
  `avreymi.jsonl`/`ofek.jsonl` and pool, or switch to the existing
  `human_validation_results.py`/`human_validation_figures.py` kappa pipeline once there's a
  human-human overlap worth reporting). The `--check` flag on `human_validation_results.py` tells you
  completion status per annotator at any time.
- **2026-07-31 fifth follow-up: added F3 (rubric score distributions) back, kept the E2 dumbbell cut.**
  Amit reviewed a gallery of all 12 generated candidate figures and picked `f3_rubric_distributions.png`
  (100%-stacked bars of E1 rubric scores 1-5 per stratum per dimension) as worth adding back; agreed the
  E2 dumbbell plot (`f6_paired_repair.png`) should stay cut since its actual chart is just two overlapping
  dots near the ceiling by construction, with the real signal living in the side-text annotations rather
  than the visual. Copied `f3_rubric_distributions.png` into `paper/figures/`, added it as new Figure 3
  right after the E1 effect-size figure (`width=0.74\textwidth`, `\label{fig:rubricdist}`), and added one
  forward-pointing sentence + citation to it at the end of the E1 paragraph in Results ("These are shape
  shifts, not just lower averages..."). This pushed main-text length over 3 pages by ~5 lines, so trimmed
  two nearby sentences (the "E1 and E2/E3 answer different questions" transition and the closing
  Limitations caveat paragraph) to compensate — main text is back to exactly 3 pages. Appendix A now has 7
  figures/tables across 3 pages (was 2); Figure 3 no longer fits alongside Figures 1/2/Table 1 on the first
  appendix page even at a smaller width, so it starts page 2 of the appendix together with the transition
  heatmap and win-rate bar, pushing the F9a spot-check (now Figure 6) alone onto a mostly-blank page 3 of
  the appendix — accepted as a reasonable cost of one more figure rather than over-shrinking images.
  Renumbered figure references throughout: Figure 6 = F9a spot-check (was 5), Figure 7 = lead-bias
  supplementary check (was 6). Paper is now **8 pages** total (3 main + 3 Appendix A + 1 Appendix B +
  1 references).
- **2026-07-31 sixth follow-up: tightened Limitations and Conclusion (was repetitive).** Amit flagged that
  the section re-listed E1/E2/E3 findings almost verbatim right after the Results section had just stated
  them, and the two paragraphs both circled back to "reference auditing precedes fine-tuning" as a
  bookend. Rewrote to state the implication once (quality varies systematically + repair helps, per E1/
  E2/E3, without re-deriving each number) and moved straight to the two caveats and the closing call to
  action, cutting the section from two dense paragraphs to a leaner pair. Main text still ends cleanly on
  page 3 with some spare room in the last column; total page count unchanged at 8.
- **2026-07-31 seventh follow-up: merged Figure 6 onto the same page as Appendix B (was a wasted page).**
  Amit asked for the "B Supplementary Lead-Bias Check" heading and its figure to share a page with the
  F9a spot-check that preceded it (Figure 6 was sitting alone on a nearly-blank page). Root cause of why
  this hadn't been done already: simply deleting the `\clearpage` before `\section{Supplementary
  Lead-Bias Check}` reproduces the exact broken-ordering regression noted in the third follow-up above —
  with several `[!htb]` floats queued in Appendix A, LaTeX can delay a float (Figure 6) past the point
  where it's declared, but the plain-text `\section` heading right after it is not a float and keeps
  flowing normally, so it rendered on page 4, two pages before Figure 6 itself ever appeared. Tried
  `\FloatBarrier` (`placeins` package) first — it fixes the ordering (nothing appears out of sequence
  anymore) but doesn't merge the pages, since the barrier only guarantees floats resolve before the
  barrier, not that the following text fits on the same physical page. Fix that actually worked: swapped
  every appendix `[!htb]` to `[H]` (`float` package, `\usepackage{float}`) for all figures and Table 1.
  `[H]` disables floating outright — each figure/table is placed exactly where it's written in the source
  and only overflows to a fresh page if it truly doesn't fit, so order can never invert and pages pack as
  tightly as the content allows. Result: Figure 6, the "B Supplementary Lead-Bias Check" heading, and
  Figure 7 now all sit on one page, correct order preserved; total paper dropped from 8 pages back to
  **7 pages** (3 main + 2 Appendix A + [merged] Appendix B + 1 references). If more figures get added to
  Appendix A in the future, re-check for the same ordering bug before assuming `\clearpage` removal is
  safe — `[H]` placements are the reason it's safe now.
- **2026-07-31 eighth follow-up: pulled References onto the same page (was its own near-empty page).**
  Amit asked to move the references higher too. Removed the `\clearpage` before `\bibliography{bib}` the
  same way as the previous follow-up; the 8-entry list initially split awkwardly (4-6 entries on the
  Appendix-B page, 2-3 spilling onto an otherwise-blank final page). Closed the gap by: shaving
  `f5_length_lead_bias.png` from `0.8\textwidth` to `0.7\textwidth` (matches the other appendix figures'
  width band), adding `\vspace{-0.6em}` before both the "Supplementary Lead-Bias Check" heading and the
  bibliography, tightening `\bibsep` to `1pt`, and wrapping the bibliography in `\small`. Result: Figure 6
  (F9a), Appendix B (heading + Figure 7), and the complete 8-entry reference list all now fit on one final
  page with no overflow — total paper is **6 pages** (3 main + 2 Appendix A + 1 page holding Appendix B
  and References). Verified pages 1-5 are unaffected and reference text is still comfortably legible at
  `\small`.

## 2026-07-31 three-page rewrite

- **Narrative:** one audit-first story: HeSum's scraped targets require validation; deterministic and
  model-assisted curation produces 5,854 retained rows; independent E1/E2/E3 tests show the defects are
  real and the rewrites improve them.
- **Main-text structure:** Abstract; Introduction; Data; Methods; Results; Discussion; Limitations and
  Conclusion. The conclusion ends on page 3, followed by an explicit `\clearpage`.
- **Qwen:** removed completely. It no longer appears in the abstract, Introduction, Methods, Results,
  appendix, or bibliography.
- **E4:** kept modular. Methods records the controlled Arm-A/Arm-B LoRA design and required
  hyperparameters. Results contains a short reserved paragraph that currently makes no training claim.
  Replace only that paragraph when both arms finish.
- **Lead bias / S1:** the body says only that S1 cannot be compared directly with S0 because the hard
  token threshold removes common support in length, then points to Appendix Figure 6. Appendix B
  contains only F5 and a short caption.
- **Visual package:** F1 curation funnel + Table 1; F4 E1 effect sizes; F6 transition heatmap; F7 blind
  preference/placebo; sx16 test--retest kappa; F5 supplementary lead-bias check. Removed from the paper:
  F3 distributions, F6 dumbbell, F9 partial zero-shot comparison, filter overlap, edit subtypes,
  dual-reference probe prose, E4 placeholder box, and sx17 partial checkpoint comparison.
- **Build:** `main.tex` now loads `ACL2023` with `[final,nohyperref]`; `nohyperref` avoids the known
  Tectonic/XeTeX crash. Build from `paper/` with `tectonic main.tex`.

## Figures — how they're built (2026-07-29)

`paper/` and its figures are now tracked in git. Historically, the figures looked "missing" because
the plotting code and images had not been committed. The obsidian notes
(`docs/obsidian/Experiment Results.md`) referenced
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

The old F8 placeholder was removed in the three-page rewrite. Add a real E4 figure only after both
controlled training arms complete.

## Historical interim fine-tuned-vs-zero-shot artifact (removed from paper)

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
"Interim fine-tuned vs. zero-shot comparison" (`app:finetuned-interim`). The 2026-07-31 rewrite removed
this appendix because it is not the controlled E4 comparison. The historical finding was: fine-tuned
scores *lower* than zero-shot base on judge
faithfulness/fluency across every edit sub-type and both strata — a caution sign, possibly an
under-trained checkpoint (94% of one epoch, interrupted). Also corrected the E4 results text's
stale "50 of 586 test-set predictions" to "580 of 586" to match this artifact.

## Title & authors

- **Title:** Auditing HeSum: Reference Quality Defects in a Hebrew News Summarization Corpus
- **Authors:** Amit Benbenishti, Avraham Asraf, Ofek Varona

## Historical project arc (context only; not the current paper narrative)

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
- **F9 (historical side result, removed from compact paper):** Partial DictaLM2 zero-shot baseline
  (n≈604); S2 single-focus reversal.

## Key repo docs for writing

- `docs/obsidian/Experiment Results.md` — article-ready numbers
- `docs/obsidian/Paper Figures.md` — F1–F9 figure claims
- `docs/obsidian/Data Curation Pipeline.md` — pipeline counts
- `docs/obsidian/Reference Quality Rubric.md` — four judge dimensions
- `docs/superpowers/specs/2026-07-26-dataset-review-experimental-design.md` — pre-registered design

## Current abstract (plain text)

HeSum is one of the few corpora large enough to train modern Hebrew news summarizers, but its targets
are extended subheadings scraped from news sites rather than summaries written for modeling. We audit
whether those targets are suitable references before treating them as ground truth. A reproducible
pipeline filters unusable rows and rewrites defective headlines; an independent rubric judge then
scores original and curated headlines directly against their articles for faithfulness, single-focus,
informativeness, and cleanliness. The audit removes 41.5% of the source corpus and rewrites 52.4% of
the retained headlines. Flagged references show large, defect-specific quality losses, while curated
headlines are preferred in 73.6% of blind comparisons; unchanged placebo pairs tie 99.5% of the time.
These results show that scraped reference quality is a first-order experimental variable, not a fixed
property of a benchmark. We release the resulting audit and frame target validation as a prerequisite
for trustworthy Hebrew summarization training and evaluation.

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
- **Document-wide spacing fix (2026-07-29):** added `\frenchspacing` to the preamble. Flagged by Amit
  on the "What is new here" paragraph specifically, but it was a document-wide default-LaTeX behavior
  (extra stretchable space after any period it guesses ends a sentence), most visible once text is
  fully justified in the narrow two-column layout. `\frenchspacing` makes all inter-word spacing
  uniform; verified visually on the rendered PDF (page 1) that the wide gaps after "second.", "string.",
  "comparison." are gone.
- **Page 1 blank-column fix (2026-07-29):** separate issue Amit flagged as "still spaces" after the
  `\frenchspacing` fix — the bottom third of page 1's right column was blank because
  `\subsection{From a model question...}` (§1.1) couldn't fit enough body lines below it at the bottom
  of that column, so LaTeX pushed the whole heading+paragraph to page 2 rather than leave an orphaned
  line. Fixed with `\enlargethispage{3\baselineskip}` right before that `\subsection`, which gives the
  column just enough extra room for the heading plus its first couple of lines. **Caveat:** this is a
  positional patch tied to the current amount of text above it — if the four opening paragraphs (or the
  abstract, or margins) change length again, this may need to be re-tuned or removed (try commenting it
  out first and re-checking page 1 before assuming it's still needed).
- **Abstract width (2026-07-29, round 3):** the earlier `\geometry{left=1.8cm,right=1.8cm}` widened
  the whole page, but the `abstract` environment in `acl2023.sty` *also* insets its own 0.6cm
  left/right margin on top of that (a "block quote" look baked into the vendor style), which is why
  the abstract still read narrow after the page-wide fix. Overrode the `abstract` environment in
  `main.tex` (wrapped in `\makeatletter`/`\makeatother`, since `\@setsize` needs `@` to be a letter,
  which is only automatic inside a `.sty`) to shrink that inset to 0.1cm, so the abstract now uses
  almost the full column width, matching the body text below it.
- **Merged Era 2 + Era 3 into one "Era 2" (2026-07-29, §1.1 pivot narrative):** Amit pointed out that
  switching to DictaLM2.0 (old Era 2) wasn't itself a new finding — it's Hebrew-native, expected to be
  better-suited and better-tokenized for Hebrew than Qwen, so the real content of that era was always
  the pivot to data (old Era 3). Merged them into a single "Era 2 -- DictaLM2.0 and the pivot to data"
  paragraph: base-model switch explicitly framed as expected/not-a-finding, then the actual finding
  (zero-shot beats reference) and the HeSum-paper explanation, in one flow. Now exactly two named eras
  (  matches the "Two findings redirected it" transition sentence, which needed no change). No numbering
  elsewhere in the paper referenced "Era 3" so nothing else needed updating.
- **Data section tightened (2026-07-29, §2, all four subsections):** per the assignment rubric ("specify
  the data... data statistics, is it synthetic/real world, reference the original paper... use the
  format of Table 1"): made the real-world/not-synthetic claim explicit in §2.1 (previously only implied
  by "scraped from news outlets"); trimmed passive/redundant phrasing throughout §2.2's curation-pipeline
  list, §2.3's taxonomy paragraph, and §2.4's train/test-boundary paragraph. All existing statistics,
  the table (Table 1, already in the rubric's requested format), and both figures (F1, F2) were kept
  unchanged — this was a prose-precision pass only, no numbers or structure changed.

## Decisions made this round (appendix removal + Table 1 caption fix, 2026-07-29)

- **Table 1 caption "(see below)" clarified:** Amit asked where that pointer actually resolves. It's
  the `\paragraph{Why article length is a covariate, not a stratum.}` text right after Figure 2
  (`fig:prevalence`) in §2.3 — explains that S1 (`over-token-budget`) is excluded as a stratum because
  it has no common support with S0 in article length. "(see below)" was vague (depends on where LaTeX
  floats things, not guaranteed adjacent on the page), so reworded the caption to point at
  `Figure~\ref{fig:prevalence}` by name instead — resolves correctly regardless of float placement.
- **Removed Appendix "Qwen3-2B fine-tuning history" (`app:qwen`/`tab:qwen`) entirely,** per Amit's
  explicit request. Deleted the section and its v1/v2/v3 ROUGE/BERTScore table. Two dangling
  `\ref{app:qwen}` had to be fixed so compilation wouldn't produce `??`: the Era 1 paragraph's "(Appendix
  ...)" parenthetical was dropped (the "three training iterations" claim reads fine standalone — it's
  already unpacked in the next two sentences), and the Baselines/triangulation paragraph's "demoted to
  an appendix comparability table (Appendix ...)" was reworded to "demoted to a supporting role" since
  there's no longer a dedicated ROUGE/BERTScore comparability appendix to point to (ROUGE/BERTScore
  still show up elsewhere — Appendix "Dual-reference zero-shot probe" and the interim
  fine-tuned-vs-zero-shot appendix). Paper is now 9 pages (was 10). Verified via PyMuPDF text extraction
  that "Qwen3-2B fine-tuning history" no longer appears anywhere and there are no orphaned `??`.
- **Removed Figure 2 (`fig:prevalence`, `f2_defect_prevalence.png`) as redundant with Table 1,** per
  Amit. F2 was a horizontal bar chart of the same S0/S2/S3/S4/S5 stratum counts already in Table 1's
  rows, plus one extra detail (S5 broken into its 3 sub-labels) that isn't load-bearing for any claim in
  the text. Removing it meant: dropping the `Figure~\ref{fig:prevalence}` cross-ref in the §2.3 opening
  sentence (now just cites Table 1), and rewriting Table 1's caption to state the S1-skip reason inline
  instead of pointing at the (now-gone) figure — this also fully resolves last round's "where does 'see
  below' point" question, since there's no longer a figure to get the pointer tangled around. The
  `\paragraph{Why article length is a covariate...}` explanation paragraph is unchanged and still
  immediately follows the table. `figures/f2_defect_prevalence.png` and its generator
  (`data_curation/analysis/figures.py::build_f2_defect_prevalence`) were left in place (still a valid
  analysis artifact) — only the paper's citation of it was removed. Paper is now 9 pages, 12 figures
  cited.
- **Trimmed "Why article length is a covariate, not a stratum" (2026-07-29):** Amit asked for an opinion
  on the lead-bias forward-reference in this paragraph and was considering cutting it. Agreed it should
  be cut down: its statistical justification (no common support between S0 and the over-token-budget
  rows, so any comparison collapses onto a length effect) had just become a duplicate of Table 1's
  caption (added last round). Removed that restated justification but kept a single sentence pointing
  forward to the continuous-length treatment in Results (§`sec:results`, Figure~`fig:f5`) — this is the
  part that isn't redundant, since lead-bias is a named thread from the Introduction's pivot narrative
  and cutting the pointer entirely would leave that thread dangling between Data and Results.
- **Switched `[review]` to `[final]` in the `ACL2023` package option (2026-07-29):** Amit asked to
  "remove the numbering of the rows" — those were ACL review-mode margin line numbers (`lineno` package,
  enabled by the `review` option per `acl2023.sty`'s own comment "Remove the review option to generate
  the final version"), which had been leaking into every pasted PDF quote this whole session (e.g. "95
  Why article length...", "05 reduce the spaces"). `final` is `acl2023.sty`'s default and, since the
  `\author{}` block already has real names (not an anonymous placeholder), there was no blind-review
  reason to keep `review` mode. Two side effects worth knowing: author names now render under the title
  (review mode's `\outauthor` suppresses them for blind review), and page numbers are gone (final-copy
  convention — `\thispagestyle{empty}`). The placeholder `<email>` addresses are now visible in the
  compiled PDF, which raises the priority of that still-pending task.

**Methods (§3) revised (2026-07-29).** Per the assignment rubric ("if you trained a model, describe the
model architecture, the training method — loss functions, hyperparameter choice — and evaluation
method"), the section previously jumped straight into the rubric-judge instrument (evaluation
methodology) without ever describing the actual DictaLM2 fine-tuning recipe in one place — only a single
sentence in E4 mentioned LoRA r/alpha in passing, with no architecture, no loss function, no epochs/LR,
no decode config. Added a new `\subsection{Model and fine-tuning setup}` (§3.1, first in Methods, before
the evaluation-instrument subsections) covering: base model architecture (Mistral-7B, Hebrew-native,
why chosen over Qwen3-2B, cross-referenced to §1.1's pivot narrative), LoRA hyperparameters and *why*
MLP projections are included alongside attention (attention-only degenerates into lead-copying —
deliberately tied to the paper's lead-bias thread), the loss function (`completion_only_loss`, i.e.
cross-entropy masked to the completion), full training hyperparameters (1 epoch, LR 2e-4 cosine, 5%
warmup, bf16, effective batch 16), and inference/decode config (greedy, 128-token budget, Hebrew-script
decode constraint). Added a `hu2022lora` BibTeX entry (Hu et al. 2022, ICLR) for the LoRA citation.
Trimmed E4's own methodology paragraph to point back at this new subsection instead of repeating the
same hyperparameters. Clustering/other-procedures rubric clause doesn't apply — topic clustering is
side infrastructure, not part of the paper's reported experiments. Compiles clean, 9 pages, citation
resolves correctly.

**Fixed a bad line-break in §3.1 (2026-07-29).** Amit flagged "something is wrong with the rendering of
3.1" — the opening line ("E4 (below) fine-tunes dicta-il/dictalm2.0-instruct...") had huge stretched gaps
between words. Root cause: `\texttt` (monospace) disables hyphenation, so the ~35-character model-id
token is one unbreakable unit; when it doesn't fit the line remainder, LaTeX shoves the whole token to
the next line and stretches the sparse leftover words to fill the column width (a very underfull hbox).
This is a general risk for every long `\texttt{...}` model identifier in a narrow two-column layout, not
just this one occurrence — the same string appears 8 times across the paper. Fixed at the source instead
of patching this one instance: added `\dictalmtwo`, `\dictalmthreesmall`, `\qwenthreetwob` macros in the
preamble that insert `\allowbreak` at each existing hyphen/slash inside the identifier (no visible
change, just legal break points), and replaced all raw `\texttt{dicta-il/dictalm2.0-instruct}` /
`\texttt{DictaLM-3.0-1.7B-Instruct}` / `\texttt{Qwen/Qwen3-2B}` occurrences with the macros. Verified by
rendering every page that mentions a model name (0, 1, 2, 3, 5, 7) — all wrap cleanly now. Shorter
`\texttt` identifiers (`gemini-2.5-flash-lite`, `completion_only_loss`, the edit-type labels) were left
as plain `\texttt` since they're short enough not to trigger this in practice (confirmed visually, no
bad breaks) — if one ever does, the fix is the same `\allowbreak` pattern.

## Open items for abstract

- [ ] Fix author emails in `main.tex` (still placeholder `<email>`)
- [x] Re-read the new narrative abstract vs. the Introduction's pivot subsection (§1.1) — some overlap
      is expected and fine (abstract compresses, §1.1 elaborates with the three named eras and specific
      numbers 2.98/2.64); no rewording needed.

## Amit's remarks (paste below)

<!-- Add your abstract feedback here -->

## To check yourself (2026-07-29)

- [ ] **Lead-bias framing.** You've trimmed/questioned this thread twice already (the "Why article
      length is a covariate" paragraph in §2.3, and the forward-pointer to Results). Read it end to end
      once: Introduction §1 (original motivating question) → §2.3's one-sentence pointer → §4.2 "Length
      and lead bias are continuous, not filter artifacts" (the actual F5 payoff) → §3.1's new mention
      ("the same shortcut this project's lead-bias question is about," re: LoRA target modules). Decide
      if that's the right amount of emphasis/repetition or if it should be trimmed further or
      consolidated — this is a judgment call the agent can't make for you.
- [ ] Re-read Methods (§3) in full now that §3.1 is new — you flagged the §3.1 render bug, worth
      double-checking the rest of §3.1's content/wording (not just layout) reads the way you want before
      moving on to Results (§4).
