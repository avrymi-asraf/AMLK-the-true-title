"""
The prompt-optimization loop's round registry: every set of prompt candidates ever tried.

Role: the loop's *input*, and half of its permanent record — `docs/prompt-arena-notebook.md`
is the other half (what happened and why). Each round states the hypothesis it tests and
lists short candidates; a round is never edited after it runs, so the file reads as the
experiment's history and the diff between rounds shows exactly what changed.

Code flow: ROUNDS[n] -> prompt_arena.write_round -> prompt_sweep_hf_job (GPU) ->
prompt_arena scoring -> notebook entry -> a new ROUNDS[n+1] here.

Execution environment: local, pure data — no GPU, no API, no imports beyond the dataclass.
"""
from data.prompts import PROMPT_TEMPLATE
from evaluation.prompt_arena import PromptCandidate

# Round 1 — baseline sweep. Hypothesis: the prompt's *language* (Hebrew vs English) and
# whether it spells out the anti-digest rule both matter for a Hebrew instruct model.
# p0_current is the control: the hardened prompt currently in data/prompts.py.
ROUND_1_HYPOTHESIS = (
    "Does a short Hebrew instruction beat the long English hardened prompt, and does "
    "explicitly banning lists/pipes earn its tokens?"
)

ROUND_1 = [
    PromptCandidate(
        id="p0_current",
        template=PROMPT_TEMPLATE,
        note="control: the hardened prompt in data/prompts.py (long, English)",
    ),
    PromptCandidate(
        id="p1_he_minimal",
        template="סכם את כתבת החדשות הבאה במשפט אחד קצר ועובדתי בעברית.\n\n{text}\n\nתקציר:",
        note="shortest Hebrew instruction, one sentence",
    ),
    PromptCandidate(
        id="p2_he_two_sent",
        template=(
            "סכם את כתבת החדשות הבאה במשפט אחד או שניים בעברית, על סמך הכתבה בלבד.\n\n"
            "{text}\n\nתקציר:"
        ),
        note="Hebrew, 1-2 sentences + grounding clause",
    ),
    PromptCandidate(
        id="p3_he_no_lists",
        template=(
            "סכם את כתבת החדשות הבאה במשפט אחד או שניים בעברית. כתוב פרוזה רציפה, "
            "בלי רשימות ובלי הסימן '|'.\n\n{text}\n\nתקציר:"
        ),
        note="Hebrew + explicit anti-digest rule (does banning lists earn its tokens?)",
    ),
    PromptCandidate(
        id="p4_en_short",
        template=(
            "Summarize this Hebrew news article in one short factual Hebrew sentence.\n\n"
            "{text}\n\nSummary:"
        ),
        note="short English instruction — isolates prompt language from prompt length",
    ),
]

# The 2-candidate slice used for the cheap smoke run (verifies the pipeline, not the prompts).
SMOKE = ROUND_1[:2]

# Round 2 — word-budget sweep. Round 1 (100 examples/prompt) found two things:
# (a) sentence-count phrasing ("one or two sentences") does not bind length — all 5 round-1
#     prompts clustered at 52-65 words / 3.4-4.2 sentences regardless of wording, all hitting
#     generation's max_new_tokens cap;
# (b) the long English "Rules:"-bulleted prompt (p0_current) triggered a garbled Hangul
#     near-token ("[/인스트]", an apparent hallucinated echo of Mistral's own [/INST] closing
#     tag) in 38% of its outputs, vs 11% for a short English prompt and 1-2% for Hebrew
#     prompts — fixed at the decode-constraint level (evaluation/hebrew_constraint.py now also
#     bans CJK/Hangul) but the prompt-language effect itself is a real, transferable finding.
# Round 2 tests whether a stated numeric word budget succeeds where sentence-count phrasing
# failed, holding max_new_tokens at 160 (unchanged) so a tight cap can't be mistaken for a
# prompt actually binding length.
ROUND_2_HYPOTHESIS = (
    "Round 1: sentence-count phrasing ('one or two sentences') does not bind length — all 5 "
    "prompts overshot to 52-65 words / 3.4+ sentences. Does an explicit numeric word budget "
    "(e.g. 'up to 25 words') bind length where sentence-count phrasing did not, holding "
    "max_new_tokens=160 fixed so the cap can't be confused for the prompt working?"
)

ROUND_2 = [
    PromptCandidate(
        id="p1_he_minimal",
        template="סכם את כתבת החדשות הבאה במשפט אחד קצר ועובדתי בעברית.\n\n{text}\n\nתקציר:",
        note="round-1 winner, unchanged — control to isolate the word-budget effect",
    ),
    PromptCandidate(
        id="p5_he_wordcap25",
        template=(
            "סכם את כתבת החדשות הבאה בעברית, במשפט אחד בלבד ובאורך של עד 25 מילים.\n\n"
            "{text}\n\nתקציר (עד 25 מילים):"
        ),
        note="explicit numeric word cap (25), stated twice (instruction + label)",
    ),
    PromptCandidate(
        id="p6_he_wordcap15",
        template=(
            "סכם את כתבת החדשות הבאה בעברית במשפט קצר אחד, לא יותר מ-15 מילים.\n\n"
            "{text}\n\nתקציר (עד 15 מילים):"
        ),
        note="tighter numeric cap (15) — does a smaller number bind harder?",
    ),
    PromptCandidate(
        id="p7_he_wordcap_grounded",
        template=(
            "סכם את כתבת החדשות הבאה בעברית, במשפט אחד או שניים ובאורך של עד 30 מילים, "
            "רק על סמך המידע בכתבה, בלי רשימות ובלי הסימן '|'.\n\n"
            "{text}\n\nתקציר (עד 30 מילים):"
        ),
        note="word cap (30) + grounding + anti-digest, combined — the round-1 lessons stacked",
    ),
    PromptCandidate(
        id="p8_en_wordcap",
        template=(
            "Summarize this Hebrew news article in one Hebrew sentence of at most 25 words.\n\n"
            "{text}\n\nSummary (max 25 words):"
        ),
        note="English + numeric cap — isolates whether the cap works regardless of language",
    ),
]


# Round 3 (final, per the 3-round budget) — one-shot exemplar sweep. Round 2 found that a
# numeric word budget only weakly binds length: every capped prompt overshoot its own stated
# number by 2-3x (p6 asked for <=15 words, delivered 47.3), and 15/25/30 barely separated
# output length from each other. The model is not doing arithmetic against a stated number.
# Round 3 tests the strongest untested lever: a worked example (one/two-shot) showing the
# exact target length and form, instead of describing it abstractly. max_new_tokens stays at
# 160 (unchanged across all 3 rounds) so cap and wording are never confounded.
ROUND_3_HYPOTHESIS = (
    "Round 2: a numeric word budget only weakly binds length (stated caps 15/25/30 produced "
    "47-54 words, barely separated from each other or from the uncapped 58-word control). Does "
    "a worked example (one-shot or two-shot) of the exact target length/form bind length where "
    "abstract numeric instructions did not?"
)

_EXAMPLE_ARTICLE_1 = (
    "משטרת ישראל עצרה אתמול שני חשודים בפריצה לדירה בתל אביב, לאחר שנתפסו בסמוך למקום עם "
    "רכוש גנוב. השניים נחקרים בתחנת מרכז העיר."
)
_EXAMPLE_SUMMARY_1 = "המשטרה עצרה שני חשודים בפריצה לדירה בתל אביב עם רכוש גנוב."

_EXAMPLE_ARTICLE_2 = (
    "עיריית חיפה הודיעה על תוכנית לשיפוץ שלושה גנים ציבוריים בעיר במהלך השנה הקרובה, "
    "בעלות כוללת של כ-5 מיליון שקל. העבודות צפויות להתחיל בחודש הבא."
)
_EXAMPLE_SUMMARY_2 = "עיריית חיפה תשפץ שלושה גנים ציבוריים בעלות של כ-5 מיליון שקל."

ROUND_3 = [
    PromptCandidate(
        id="p6_he_wordcap15",
        template=(
            "סכם את כתבת החדשות הבאה בעברית במשפט קצר אחד, לא יותר מ-15 מילים.\n\n"
            "{text}\n\nתקציר (עד 15 מילים):"
        ),
        note="round-2 winner, unchanged — control to isolate the exemplar effect",
    ),
    PromptCandidate(
        id="p9_he_oneshot",
        template=(
            "להלן דוגמה לתקציר טוב: משפט אחד קצר, עובדתי, עד 15 מילים.\n\n"
            f"כתבה לדוגמה: {_EXAMPLE_ARTICLE_1}\n"
            f"תקציר לדוגמה: {_EXAMPLE_SUMMARY_1}\n\n"
            "כעת סכם באותו סגנון בדיוק את הכתבה הבאה, במשפט אחד ועד 15 מילים בלבד:\n\n"
            "{text}\n\nתקציר (עד 15 מילים):"
        ),
        note="one worked example (article+summary pair) before the real article",
    ),
    PromptCandidate(
        id="p10_he_twoshot",
        template=(
            "להלן שתי דוגמאות לתקציר טוב: משפט אחד קצר, עובדתי, עד 15 מילים.\n\n"
            f"כתבה לדוגמה 1: {_EXAMPLE_ARTICLE_1}\n"
            f"תקציר לדוגמה 1: {_EXAMPLE_SUMMARY_1}\n\n"
            f"כתבה לדוגמה 2: {_EXAMPLE_ARTICLE_2}\n"
            f"תקציר לדוגמה 2: {_EXAMPLE_SUMMARY_2}\n\n"
            "כעת סכם באותו סגנון בדיוק את הכתבה הבאה, במשפט אחד ועד 15 מילים בלבד:\n\n"
            "{text}\n\nתקציר (עד 15 מילים):"
        ),
        note="two worked examples — does reinforcing the pattern twice help further?",
    ),
    PromptCandidate(
        id="p11_he_stopcue",
        template=(
            "סכם את כתבת החדשות הבאה בעברית במשפט קצר אחד, לא יותר מ-15 מילים. "
            "כתוב משפט אחד בלבד ועצור מיד בסופו.\n\n"
            "{text}\n\nתקציר (משפט אחד, עד 15 מילים):"
        ),
        note="explicit stop-after-one-sentence cue, no exemplar — isolates cue from example",
    ),
    PromptCandidate(
        id="p12_he_oneshot_grounded",
        template=(
            "להלן דוגמה לתקציר טוב: משפט אחד קצר, עובדתי, עד 15 מילים, המבוסס רק על הכתבה.\n\n"
            f"כתבה לדוגמה: {_EXAMPLE_ARTICLE_1}\n"
            f"תקציר לדוגמה: {_EXAMPLE_SUMMARY_1}\n\n"
            "כעת סכם באותו סגנון בדיוק את הכתבה הבאה, במשפט אחד ועד 15 מילים בלבד, "
            "רק על סמך המידע שבה:\n\n"
            "{text}\n\nתקציר (עד 15 מילים):"
        ),
        note="one-shot + explicit grounding clause — the strongest lever combo tested so far",
    ),
]

ROUNDS: dict[int, list[PromptCandidate]] = {
    1: ROUND_1,
    2: ROUND_2,
    3: ROUND_3,
}

HYPOTHESES: dict[int, str] = {
    1: ROUND_1_HYPOTHESIS,
    2: ROUND_2_HYPOTHESIS,
    3: ROUND_3_HYPOTHESIS,
}


def get_round(round_num: int, smoke: bool = False) -> list[PromptCandidate]:
    if smoke:
        return SMOKE
    if round_num not in ROUNDS:
        raise KeyError(
            f"round {round_num} is not defined — add it to ROUNDS in evaluation/prompt_rounds.py "
            f"(defined: {sorted(ROUNDS)})"
        )
    return ROUNDS[round_num]
