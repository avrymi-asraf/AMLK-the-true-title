ANLP Research Proposal (revised)

AMLK - Fine-tuning for Hebrew news summarization

Amit Benbenishti, Avraham Asraf

This revision reworks the proposal in response to reviewer feedback. The main change is the
positional-shortcut probe: instead of training a separate model per input variant (which answers
"how learnable is the reference from each slice of the article" - a question about the data), the
probe now trains a single model on whole articles and ablates the input at inference time (which
answers "what does the trained model rely on" - the question we actually care about). We add a
length-matched control, restrict the probe to examples whose summary content is genuinely present
in the body, add a complementary training-distribution experiment, add a Lead-N baseline, switch
the LLM judge off the Gemini family to avoid self-preference bias, characterize abstractiveness of
the data before running, and adopt the HeSum recommendations for ROUGE in Hebrew.

Abstract - We propose AMLK, a study of instruction-tuned abstractive summarization for Hebrew news
articles, a morphologically rich, medium-resource setting under-represented in current models. We
fine-tune Qwen/Qwen3-2B on professional Hebrew article-summary pairs, compare three fine-tuning
regimes (QLoRA, LoRA, full FT), benchmark against a stronger model baseline and a simple Lead-N
baseline under the same metrics, and probe whether the trained model aggregates global context or
latches onto the lead. The shortcut probe is an inference-time ablation on a single whole-article
model, with controls that separate genuine positional reliance from input-length and
information-availability confounds. The goal is a reproducible Hebrew news-summarization baseline
with a clean diagnostic for positional shortcuts and systematic error analysis.

Existing Work - English news summarization (CNN/DM, XSum; BART, T5, PEGASUS) established that models
often exploit lead bias, that ROUGE alone misaligns with human judgment, and that stronger baselines
are needed to interpret automatic metrics - lessons we transfer to Hebrew without re-running English
experiments. Lead bias is strong enough that Lead-N is a standard, hard-to-beat baseline (Nallapati
et al., 2016; See et al., 2017), and faithfulness failures are common even at high ROUGE (Maynez et
al., 2020). LLM-as-judge evaluation is convenient but biased toward outputs from the same model
family (self-preference; Panickssery et al., 2024), which motivates judging with a model outside the
baseline's family. Hebrew-specific work has focused on benchmarks: HeSum (Paz-Argaman et al., 2024)
and IAHLT summarization_he provide professional news article-summary pairs; HeSum also documents the
limits of ROUGE under Hebrew morphology and recommends morphological normalization before scoring.
LoRA and QLoRA make 2-7B LLM fine-tuning feasible on a single GPU.

Models - We use Qwen/Qwen3-2B as the primary model. Under identical data and hyperparameters we
compare QLoRA (~8 GB VRAM), LoRA bf16 (~16 GB), and full FT on HuggingFace Jobs. We report two
reference points: (1) a simple extractive Lead-N baseline (first N sentences as the summary), the
standard lead-bias lower bound for news; and (2) an advanced-model baseline - a stronger model
(e.g. Gemini via API, or a larger open LLM) on the same Hebrew test set with the same instruction
template - so metric scores can be read against both a trivial positional baseline and a strong
upper bound. All fine-tuned variants use the trl SFT trainer; the default prompt is "Summarize the
following Hebrew text: ... Summary: ...".

Dataset - We focus on Hebrew journalism data: HeSum (~10,000 professional news article-summary
pairs) and IAHLT summarization_he, normalized to {text, summary, source} and split 80/10/10
stratified by source. Before any training we characterize the data: (a) abstractiveness - novel
n-gram rate of the summary vs the article, extractive-fragment coverage/density (Grusky et al.,
2018), and a small manual check of how many summaries are genuinely abstractive vs near-extractive;
(b) lead alignment - summary-vs-lead overlap (ROUGE and longest-common-substring) and
summary-vs-body overlap, to quantify how lead-aligned the gold summaries are and to define the
subsets used by the probe below. Reference summaries in news corpora are often headline-like (short,
lead-aligned); we optionally vary the instruction to control output length/style (e.g. one-line
headline vs multi-sentence summary) and measure the effect with the same metric battery.

Evaluation - All systems are scored with ROUGE-1/2/L, BERTScore (xlm-roberta-large), and an
LLM-as-judge for faithfulness and fluency (1-5). For ROUGE we follow the HeSum recommendations for
Hebrew (morphological normalization / lemmatization before matching) so that inflected forms are not
counted as misses, and we report both the raw and normalized variants. To avoid self-preference
bias, the LLM judge is from a different model family than the advanced baseline: if Gemini is the
advanced baseline, the judge is a non-Gemini model (e.g. GPT-4-class or Claude), and vice versa.
Error analysis: sample ~50-100 test predictions and label failure types reported in the
summarization literature (hallucination, omission, entity/number errors, lead copying, fluency),
reporting rates by model - following the finding that published systems often fail on faithfulness
despite high ROUGE.

Main experiment - Three fine-tuning regimes vs zero-shot Qwen3-2B, vs the Lead-N baseline, and vs
the advanced-model baseline, all under the metric battery above.

Positional-shortcut probe (inference-time ablation) - We train a single model on whole articles and
then ablate the input at inference, so the diagnostic targets what the trained model relies on
rather than what is learnable from a given slice. For each test article we evaluate the same model
on three inputs: Whole, Lead-only, and Body-only. A large drop on Body-only together with little
drop on Lead-only is evidence that the model leans on the lead rather than aggregating the body.
Two controls keep the comparison clean:
  - Information availability. The lead often contains the very content that the (lead-aligned) gold
    summary needs, so removing it can simply delete the answer - a human would also do worse. We
    therefore restrict the primary probe to a "body-supported" subset: examples where the gold
    summary content is genuinely present in the body (high summary-vs-body overlap), so that the
    body alone could in principle support a good summary.
  - Input length. Removing the lead also shortens the input, and length alone affects quality. We
    add a length-matched control input that removes the same number of tokens as the lead, but taken
    from a random span after the lead (so Whole-minus-lead-length and Body-with-random-cut have
    equal length). Comparing Body-only against this length-matched cut isolates the positional
    effect from the length effect.
We also confirm the premise directly: on the body-supported subset, a strong reference system (the
advanced baseline) can still produce a good summary without the lead - otherwise a Body-only drop is
uninformative.

Training-distribution experiment (what the data teaches) - Complementing the inference probe, we ask
whether the training distribution shapes lead reliance. We split the train set by summary-vs-lead
overlap (simple ROUGE) and train two whole-article models with identical hyperparameters: one on the
N examples whose summary least overlaps the lead, and one on N random examples. We then run the same
inference ablation on both. If the low-overlap-trained model relies less on the lead (smaller
Lead-only advantage, smaller Body-only drop), that shows lead reliance is partly inherited from a
lead-aligned training distribution rather than being intrinsic to the task.

Milestones -
- 27.05 - Stage A: data pipeline and training scripts; HF Jobs verified. (Done.)
- 07.06 - Stage B: metrics + advanced baseline + Lead-N baseline + error analysis on first Hebrew run.
- 14.06 - Presentation: QLoRA / LoRA / full FT vs baselines; news/journalism framing; data characterization.
- 30.06 - Positional-shortcut probe (inference ablation + controls) and training-distribution experiment complete.
- 31.07 - Final paper and presentation.

References

Paz-Argaman, T., Mondshine, I., Achi Mordechai, A., & Tsarfaty, R. (2024). HeSum: a Novel Dataset for Abstractive Text Summarization in Hebrew. Findings of ACL 2024. arXiv:2406.03897.

Nallapati, R., Zhou, B., dos Santos, C., Gulcehre, C., & Xiang, B. (2016). Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond. CoNLL.

See, A., Liu, P. J., & Manning, C. D. (2017). Get To The Point: Summarization with Pointer-Generator Networks. ACL.

Grusky, M., Naaman, M., & Artzi, Y. (2018). Newsroom: A Dataset of 1.3 Million Summaries with Diverse Extractive Strategies. NAACL.

Maynez, J., Narayan, S., Bohnet, B., & McDonald, R. (2020). On Faithfulness and Factuality in Abstractive Summarization. ACL.

Panickssery, A., Bowman, S. R., & Feng, S. (2024). LLM Evaluators Recognize and Favor Their Own Generations. NeurIPS.

Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. arXiv:2305.14314.

Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: Evaluating Text Generation with BERT. ICLR.
