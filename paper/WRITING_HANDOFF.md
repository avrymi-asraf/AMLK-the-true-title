# AMLK paper — writing handoff

**Last updated:** 2026-07-29  
**Status:** Abstract, Introduction, Data (§2), and Methods (§3) revised. Abstract (lines 43–59): narrative pivot framing,
"multi-stage" instead of a specific stage count (fixed a real off-by-one: text said "eight-stage" but the
enumerated pipeline in §3.2/Table 1 has 7 items). Introduction (§1, lines 69–158): rewritten to be more
to the point per the assignment rubric's 4 required questions (no-jargon objective, current practice +
limits, what's new + why it'll work, who cares) — each already had its own `\paragraph{}` header, so this
pass tightened prose and cut jargon rather than restructuring; also swapped one hardcoded citation
("Nallapati et al. (2016) and See et al. (2017)") for `\citet{}` so it can't drift from `bib.bib`, giving
the section both the bracket and non-bracket citation styles the rubric asked for. All figures the
paper cites are real, generated PNGs in `paper/figures/` — see "Figures" section below. Bibliography
(`bib.bib`) reconstructed and working — no more `?` for unresolved citations. Data (§2, all 4 subsections)
tightened for precision/concision (real-world-corpus statement, action-led pipeline steps, condensed
taxonomy/train-test-boundary prose) — no numbers changed. Removed Appendix "Qwen3-2B fine-tuning
history" (`app:qwen`/`tab:qwen`), and removed Figure 2 (`fig:prevalence`) as redundant with Table 1 —
paper is 9 pages, 12 figures cited (was 13). Switched `\usepackage[review]{ACL2023}` to `[final]`:
removes the margin line numbers (source of those stray digits Amit kept seeing in pasted PDF text, e.g.
"95 Why article length..."), and as a side effect now shows real author names (previously hidden by
review-mode blind-review placeholder) and drops page numbers (standard ACL final-copy behavior). Trimmed
the "Why article length is a covariate, not a stratum" paragraph — its statistical justification was
now duplicated with Table 1's caption, kept only the one-sentence forward-pointer to the lead-bias
analysis in Results. See "Decisions made this round (appendix + Table 1 caption)" below. Pending: real
author emails (still `<email>` placeholders, now visible in the compiled PDF). Methods (§3) gained a new
"Model and fine-tuning setup" subsection (architecture/loss/hyperparameters — see below). Next:
Results (§4), or whichever section you flag next.

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
