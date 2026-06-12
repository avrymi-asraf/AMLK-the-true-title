# TODO List

A. Train pipline
    A.1 Download the dataset: https://github.com/IAHLT/summarization_he, https://github.com/OnlpLab/HeSum
    A.2 Download the pretrained model: qwen 3.5 2b
    A.3 Fine-tune the model on the dataset using Hugging Face, if the compute is not enough, run it on HF job.
B. Evaluation pipeline
    B.1 rouge evaluation
    B.2 bert score evaluation
    B.3 LLm evaluation
    B.4 Advanced-model baseline: run a stronger model (e.g. Gemini API) on the same Hebrew test set with the same prompt; score with B.1-B.3 for comparison.
    B.5 Error analysis: sample ~50-100 predictions; label failure types (hallucination, omission, wrong entity/number, lead copying, fluency); report rates per model.
C. Literature exploration - 24.05 - unordred tasks
    C.1 Survey English news summarization (datasets, models, lead bias, metric limits) and map lessons to our Hebrew setup - no English training run required.
    C.2 wriet abstract for the project
    C.3 Define goles and milestones for the project
D. Initial results
    D.1 real training
    D.2 Improve the training
E. Present the results - 14.06
    E.1 Write a paper
    E.2 Prepare a presentation
F. Truncation / positional-shortcut experiment
    F.1 Add a preprocessing step that splits each article into three input variants: Whole text (baseline), Lead-only (opening segment), Body-only (article with the lead removed).
    F.2 Train one model per variant (Whole, Lead-only, Body-only) with identical hyperparameters and number of training steps.
    F.3 Evaluate each model on its matching test variant using ROUGE / BERTScore / LLM-as-judge.
    F.4 Hypothesis: a significant accuracy drop on Body-only inputs (relative to Whole text and Lead-only) would prove that the model relies on positional shortcuts rather than global context.
G. Hebrew news / headline control (journalism focus)
    G.1 Emphasize journalism subset in analysis (HeSum + IAHLT news articles; stratify or report by source).
    G.2 Optional: train or evaluate with alternate instructions (one-line headline vs multi-sentence summary) and compare metrics.
H. Finalize the project - 31.07
