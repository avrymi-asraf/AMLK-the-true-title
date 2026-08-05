"""Prompt and structured-output schema for headline target refinement."""

TARGET_REPAIR_SCHEMA = {
    "type": "object",
    "properties": {
        "replacement_headline": {
            "type": ["string", "null"],
            "description": (
                "Null if the existing headline should be kept exactly. "
                "A replacement Hebrew headline if the existing headline should be rewritten."
            ),
        },
    },
    "required": [
        "replacement_headline",
    ],
    "additionalProperties": False,
}

PROMPT = """
# Task

Curate headline targets for a Hebrew headline-generation training dataset.

We are preparing training examples for this generation task: given a Hebrew
source_text, generate a concise, accurate, and informative Hebrew headline
that reflects the source_text's central focus.

You are given a Hebrew source_text and an existing_headline.

Your job is to decide whether the existing_headline should remain the
training target, be lightly revised, or be fully rewritten.

# Replacement Policy

Use the least invasive valid action.

Original human-written headlines are valuable, but the final headline must be a
clean gold training target. Target quality is more important than preserving the
original wording.

Return null for replacement_headline when the existing_headline should be
kept exactly as the training target. Do this when the existing_headline is
already faithful, informative, understandable, clean enough, and representative
of the source_text's central focus. Do not rewrite a usable headline merely to
improve style, wording, or elegance.

Return a Hebrew string for replacement_headline when the existing_headline
should be cleaned, lightly refined, or fully rewritten. The returned string will
replace the existing_headline as the training target.

If the existing_headline is basically valid but contains removable noise or
minor defects, prefer a cleaned or lightly refined replacement_headline. This
includes cases where the headline can be made suitable by removing boilerplate,
site labels, pipe-separated fragments, duplicated text, odd punctuation, or
minor broken formatting while preserving the original headline's core meaning.

Fully rewrite the headline only when the existing_headline is fundamentally
unsuitable as a gold training target, such as when it is misleading, unsupported
by the source_text, too vague to identify the central focus, broken beyond
light repair, or not actually a headline or headline-like text.

# Headline Quality Rules

A good headline target is a short Hebrew headline-like text that gives a
faithful, informative entry point into the source_text. It may resemble a
headline, subtitle, or extended headline. It does not need to summarize every
important detail.

A suitable headline target must:
- Be faithful to the source_text.
- Be supported by information in the source_text.
- Be understandable as natural Hebrew.
- Reflect the source_text's main story, central subject, or a salient
  article-supported angle.
- Be informative enough to serve as a useful gold training target.
- Preserve uncertainty, attribution, and framing from the source_text.
- Avoid unsupported facts, names, numbers, causes, quotes, or conclusions.

A headline may be kept even if it is imperfect. Do not replace an otherwise
usable existing_headline merely because it uses different wording, omits
secondary details, overlaps with part of the opening, is not the best possible
headline, or could be more elegant.

Do not return a replacement_headline for small local wording changes when the
existing_headline is already a usable gold training target. This includes minor
preposition choices, singular/plural nuances, small grammatical preferences,
word-order preferences, punctuation preferences, or slightly more elegant
phrasing. Keep the existing_headline unless the change fixes a problem that
materially affects its quality as a training target.

A headline is not suitable as a gold training target when it materially reduces
training quality. This includes cases where it:
- Is misleading or unsupported by the source_text.
- Focuses on the wrong topic or on a minor detail in a way that misrepresents
  what the source_text is mainly about.
- Is too vague or generic to give a useful sense of the source_text's main
  story, subject, or article-supported angle.
- Is mainly a teaser, hook, promotional sentence, call to action, or empty
  question rather than an informative headline.
- Is not actually headline-like text.
- Is badly truncated, broken, duplicated, or corrupted.
- Contains boilerplate, links, credits, metadata, site labels, category labels,
  unrelated fragments, or excessive formatting noise.
- Is only a list of pipe-separated fragments rather than one coherent headline
  target.

If the existing_headline is basically valid but contains removable noise or
minor defects, clean or lightly refine it rather than fully rewriting it.
Remove or fix boilerplate, site labels, category labels, credits, links,
duplicated text, awkward punctuation, repeated spaces, broken formatting,
unrelated fragments, or excessive separators while preserving the original
headline's core meaning as much as possible.

Fully rewrite only when the existing_headline cannot be made into a clean gold
target by light editing.

When writing replacement_headline:
- Write exactly one Hebrew headline target.
- Prefer one concise sentence, or two short sentences when needed.
- Write natural Hebrew suitable for a headline, subtitle, or extended headline.
- State the content directly instead of describing the text as a text.
- Do not begin with formulaic phrases such as "המאמר עוסק",
  "הכתבה מתארת", "הטקסט מציג", or "כותב המאמר טוען".
- For opinion, analysis, criticism, allegation, estimate, or uncertainty,
  preserve the source_text's attribution and framing.
- Do not write a formal abstractive summary, teaser, fragment list, or
  pipe-separated output.
- Do not copy a long passage from the source_text.
- Do not add unsupported background knowledge, assumptions, explanations,
  names, numbers, causes, quotes, or conclusions.

# Final Instructions

Evaluate only the supplied text. Do not infer missing content from URLs,
metadata, outside knowledge, or assumptions. Return only the structured JSON
object matching the schema. Do not explain your reasoning or add extra fields.
""".strip()
