"""Prompt and structured-output schema for source filtering."""

SOURCE_FILTER_LABELS = [
    "usable",
    "unusable_multiple_independent_items",
    "unusable_substantive_content_not_in_text",
    "unusable_damaged_or_fragmentary_text",
    "unusable_insufficient_substantive_content",
    "unusable_other",
]

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "source_filter_label": {
            "type": "string",
            "enum": SOURCE_FILTER_LABELS,
            "description": "The supplied text usability classification, including the specific category when it is unusable.",
        },
    },
    "required": [
        "source_filter_label",
    ],
    "additionalProperties": False,
}

PROMPT = """
# Task

Filter records to prepare a Hebrew training dataset for headline generation. In
the downstream task, a model will receive a source text and generate a concise,
accurate, and informative Hebrew headline that reflects the text's central focus.

Classify the supplied Hebrew text as usable for this task, or choose the
unusable category that best explains why it should be filtered out.

# Labels

Choose the single most accurate label:

- usable: the supplied text is coherent, self-contained, primarily textual, has one
  clearly dominant central focus, and contains enough substantive information
  to serve as a useful training source. It may include supporting examples,
  reactions, background, consequences, comparisons, minor subtopics, links,
  credits, calls to action, media references, formatting noise, or boilerplate
  when the essential content is present and the peripheral material can be
  safely ignored.

- unusable_multiple_independent_items: the supplied text contains two or more
  independently developed items rather than one clearly dominant subject. Use
  this when substantial parts of the text could reasonably stand as separate
  articles, including press reviews, roundups, or digests, even when the items
  share a broad person, category, publication, or theme.
  Do not use this when the text has one central focus and the additional
  material only supports, explains, illustrates, or develops that focus.

- unusable_substantive_content_not_in_text: the supplied text mainly points to,
  introduces, embeds, or surrounds substantive content that is not included in
  the supplied text. Use this when understanding the central content requires an
  unavailable video, audio recording, image, gallery, PDF, external page,
  embedded post, interactive element, or linked resource.
  Do not use this merely because the text contains a URL, media link, podcast
  link, download link, embed marker, credit, or call to action. If the supplied
  text itself contains a substantive article body, transcript, interview,
  report, or developed explanation, evaluate that text normally.

- unusable_damaged_or_fragmentary_text: the supplied text is severely corrupted,
  unintelligible, fragmentary, truncated, or assembled from broken or mismatched
  fragments in a way that prevents reliable understanding of its main content.
  Do not use this merely because the writing is imperfect, formatting is noisy,
  some sentences are awkward, minor details are missing, or the text contains ellipses.

- unusable_insufficient_substantive_content: the supplied text is coherent,
  self-contained, primarily textual, and focused on one topic, but it does not
  contain enough meaningful information, development, explanation, evidence,
  reporting, or argument to serve as a useful training source. Use this when the
  supplied text appears to be the full available text, but is too thin to
  support an informative headline.
  Do not use this merely because the text is short, introductory, or written as
  a notice. A short or introductory text is usable when it clearly identifies a
  meaningful subject, event, claim, investigation, argument, or development and
  provides enough context to support an informative headline.

- unusable_other: the supplied text is clearly unsuitable as a training source for a
  distinct reason not covered by the other labels.

# Final Instructions

Evaluate only the supplied text. Do not infer missing content from URLs,
metadata, outside knowledge, or assumptions. Choose the single most appropriate
source_filter_label. Return only the structured JSON object matching the schema.
Do not explain your reasoning or add extra fields.
""".strip()
