ANLP Research Proposal

AMLK - Fine-tuning for Hebrew news summarization

Amit Benbenishti, Avraham Asraf


Abstract - We propose AMLK, a study of instruction-tuned abstractive summarization for Hebrew news articles, a morphologically rich, medium-resource setting under-represented in current models. We fine-tune Qwen/Qwen3-2B on professional Hebrew article-summary pairs, compare three fine-tuning regimes (QLoRA, LoRA, full FT), benchmark against a stronger model baseline under the same metrics, and probe whether the model aggregates global context or latches onto the lead. The goal is a reproducible Hebrew news-summarization baseline with a diagnostic for positional shortcuts and systematic error analysis.

Existing Work - English news summarization (CNN/DM, XSum; BART, T5, PEGASUS) established that models often exploit lead bias, that ROUGE alone misaligns with human judgment, and that stronger baselines are needed to interpret automatic metrics - lessons we transfer to Hebrew without re-running English experiments. Hebrew-specific work has focused on benchmarks: HeSum (Paz-Argaman et al., 2024) and IAHLT summarization_he provide professional news article-summary pairs. LoRA and QLoRA make 2-7B LLM fine-tuning feasible on a single GPU.

Models - We use Qwen/Qwen3-2B as the primary model. Under identical data and hyperparameters we compare QLoRA (~8 GB VRAM), LoRA bf16 (~16 GB), and full FT on HuggingFace Jobs. As an advanced baseline we run a stronger model (e.g. Gemini via API, or a larger open LLM) on the same Hebrew test set with the same instruction template, so metric scores can be interpreted relative to an upper bound. All variants use the trl SFT trainer; the default prompt is "Summarize the following Hebrew text: … Summary: …".

Dataset - We focus on Hebrew journalism data: HeSum (~10,000 professional news article-summary pairs) and IAHLT summarization_he, normalized to {text, summary, source} and split 80/10/10 stratified by source. Reference summaries in news corpora are often headline-like (short, lead-aligned); we optionally vary the instruction to control output length/style (e.g. one-line headline vs multi-sentence summary) and measure the effect with the same metric battery.

Experiments - Main experiment: three fine-tuning regimes vs zero-shot Qwen3-2B and vs the advanced-model baseline, all scored with ROUGE-1/2/L, BERTScore (xlm-roberta-large), and Gemini LLM-judge (faithfulness and fluency, 1-5). Error analysis: sample ~50-100 test predictions, label failure types reported in the summarization literature (hallucination, omission, entity/number errors, lead copying, fluency), and report rates by model - following the finding that published systems often fail on faithfulness despite high ROUGE.

Truncation / positional-shortcut probe - Three input variants per article: Whole text, Lead-only, Body-only; one model per variant, evaluated on its matching test split. Hypothesis: Body-only drop (vs Whole and Lead-only) indicates reliance on positional shortcuts (the lead often overlaps with the reference summary/headline) rather than global context.

Milestones -
• 27.05 - Stage A: data pipeline and training scripts; HF Jobs verified. (Done.)
• 07.06 - Stage B: metrics + advanced baseline + error analysis on first Hebrew run.
• 14.06 - Presentation: QLoRA / LoRA / full FT vs baselines; news/journalism framing.
• 30.06 - Truncation probe complete.
• 31.07 - Final paper and presentation.


References

Paz-Argaman, T., Mondshine, I., Achi Mordechai, A., & Tsarfaty, R. (2024). HeSum: a Novel Dataset for Abstractive Text Summarization in Hebrew. Findings of ACL 2024. arXiv:2406.03897.

Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. arXiv:2305.14314.

Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: Evaluating Text Generation with BERT. ICLR.
