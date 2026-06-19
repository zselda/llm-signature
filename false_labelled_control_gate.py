import re

from LLM_models import get_default_llm_client
from Agents import prompt_lib
client = get_default_llm_client()
from datetime import datetime

# Verdict keyword patterns. Order matters at the call site: check uncertain first,
# then dissimilar, then similar ("dissimilar" contains the substring "similar").
_UNCERTAIN_PAT = re.compile(
    r"uncertain|inconclusive|unclear|indetermin|insufficient|ambiguous|"
    r"cannot (?:be )?determin|not (?:sure|certain)|in[\s-]the[\s-]middle|borderline|maybe",
    re.IGNORECASE,
)
# [ia]+ tolerates both correct ("similar") and the prompt's misspelling ("similiar").
_DISSIMILAR_PAT = re.compile(r"dissimil[ia]+r", re.IGNORECASE)
_SIMILAR_PAT = re.compile(r"simil[ia]+r", re.IGNORECASE)

# Which upstream model produced the original **dissimilar** label that the LLM is
# asked to review. Each key maps to the system prompt that names that model, so
# the LLM knows whose judgement it is re-checking. "SIAMESE" preserves the
# original behaviour; "ML" points to a generic ML-model prompt.
_FIRST_MODEL_PROMPTS = {
    "SIAMESE": prompt_lib.agent_signature_false_evaluator,
    "ML": prompt_lib.agent_signature_false_evaluator_ml,
}


def select_system_prompt(first_model="SIAMESE"):
    """Return the system prompt naming ``first_model`` as the upstream evaluator.

    ``first_model`` is matched case-insensitively against the supported sources
    (``"SIAMESE"``, ``"ML"``). Raises ValueError for any other value so a typo
    fails loudly instead of silently picking the wrong prompt.
    """
    key = str(first_model).strip().upper()
    if key not in _FIRST_MODEL_PROMPTS:
        raise ValueError(
            f"Unknown first_model {first_model!r}; "
            f"expected one of {sorted(_FIRST_MODEL_PROMPTS)}"
        )
    return _FIRST_MODEL_PROMPTS[key]


def signature_classifier(images, pagination_mode=True, log_label="", first_model="SIAMESE"):
    """Classify a signature pair as similar / dissimilar.

    ``images`` is a sequence of NumPy arrays (one (224, 224) array per signature).
    With ``pagination_mode=False`` (the false-labelled flow) the whole pair is sent
    to the model in a single multimodal request; with ``pagination_mode=True`` each
    array is sent individually.

    ``log_label`` is an optional context string (e.g. row/pair identifiers) that is
    prefixed to every printed log line so the output can be traced back to its row
    and signature pair.

    ``first_model`` selects which upstream model ("SIAMESE" or "ML") is named in
    the system prompt as the source of the original **dissimilar** label.
    """
    prefix = f"[{log_label}] " if log_label else ""
    start_time2 = datetime.now()

    SYSTEM_PROMPT = select_system_prompt(first_model)


    USER_PROMPT = """ """
    results = {}
    if pagination_mode:
        for idx, array in enumerate(images):
            start_time = datetime.now()
            print(f"{prefix}processing image {idx}")
            result = client.generate_with_image(
                model="gemma_27b",
                images=[array],
                system_prompt=SYSTEM_PROMPT,
                user_prompt=USER_PROMPT
            )

            end_time = datetime.now()

            print(f"{prefix}elapsed {end_time - start_time}")

            print(f"{prefix}{result}")
            results[idx] = result
        end_time2 = datetime.now()
        print(f"{prefix}total elapsed {end_time2 - start_time2}")

    else:
        start_time = datetime.now()
        print(f"{prefix}processing pair")
        results = client.generate_with_image(
            model="gemma_27b",
            images=images,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=USER_PROMPT
        )

        end_time = datetime.now()

        print(f"{prefix}elapsed {end_time - start_time}")

        print(f"{prefix}{results}")

    return results


def _extract_reasoning(text: str) -> str:
    """Pull the reasoning portion out of the model's free-form output."""
    marker = re.search(r"detailed_reasoning", text, re.IGNORECASE)
    if marker:
        rest = text[marker.end():]
        verdict_mark = re.search(r"verdict", rest, re.IGNORECASE)
        if verdict_mark:
            rest = rest[:verdict_mark.start()]
        return rest.strip(" \n\t{}'\":-")
    return text.strip()


def classify_verdict(raw_text):
    """Map a model response to a verdict label and its reasoning.

    Returns a ``(label, reasoning)`` tuple where label is one of
    ``"similar"``, ``"dissimilar"``, or ``"uncertain"``. The verdict is read
    from the text after the LAST occurrence of "verdict" (where the prompt asks
    the model to place it); if that is inconclusive we fall back to scanning the
    whole response. Unparseable / empty output is treated as ``"uncertain"`` so
    it gets flagged rather than silently scored.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return "uncertain", ""

    verdict_marks = list(re.finditer(r"verdict", raw_text, re.IGNORECASE))
    region = raw_text[verdict_marks[-1].end():] if verdict_marks else ""

    def _label(segment):
        if not segment or not segment.strip():
            return None
        if _UNCERTAIN_PAT.search(segment):
            return "uncertain"
        if _DISSIMILAR_PAT.search(segment):
            return "dissimilar"
        if _SIMILAR_PAT.search(segment):
            return "similar"
        return None

    label = _label(region) or _label(raw_text) or "uncertain"
    return label, _extract_reasoning(raw_text)


def compare_signature_sets(bbhs_images, talimat_images, row_id=None, first_model="SIAMESE"):
    """Compare every bbhs × talimat detection pair and aggregate to a row verdict.

    Each pair is sent to the model as exactly two images (matching the prompt's
    "2 signatures" framing). ``row_id`` is an optional identifier (e.g. the
    dataframe row index) that is included in the printed log lines together with
    the bbhs/talimat pair indices. ``first_model`` ("SIAMESE" or "ML") selects which
    upstream model is named in the prompt as the source of the **dissimilar** label.
    Aggregation:
      - llm_result = 1 (similar) if ANY pair is judged similar, else 0.
      - uncertain flag = 1 if any pair is uncertain AND no pair is similar.
    If either side has no detections, no comparison is run.

    Returns ``(llm_result, llm_reasoning, uncertain_flag, pair_results)``:
      - llm_result   : 0/1 row verdict (None when there is nothing to compare).
      - llm_reasoning: per-pair reasoning, aggregated.
      - uncertain_flag: row-level flag (see above).
      - pair_results : flat list of per-pair verdicts ordered bbhs-outer /
                       talimat-inner, encoded 1=similar, 0=dissimilar, -1=uncertain.
                       e.g. bbhs[0,1,2] x talimat[0,1]
                       -> [ (0,0),(0,1),(1,0),(1,1),(2,0),(2,1) ]  ->  [0,0,1,0,-1,0].
    """
    if not bbhs_images or not talimat_images:
        note = (
            f"No comparison run: bbhs detections={len(bbhs_images)}, "
            f"talimat detections={len(talimat_images)}."
        )
        return None, note, 0, []

    row_tag = f"row={row_id} " if row_id is not None else ""
    labels = []
    pair_results = []
    pair_reports = []
    for bi, bbhs_img in enumerate(bbhs_images):
        for ti, talimat_img in enumerate(talimat_images):
            log_label = f"{row_tag}bbhs#{bi} vs talimat#{ti}"
            raw = signature_classifier(
                [bbhs_img, talimat_img], pagination_mode=False,
                log_label=log_label, first_model=first_model,
            )
            label, reasoning = classify_verdict(raw)
            labels.append(label)
            # 1 = similar, 0 = dissimilar, -1 = uncertain
            pair_results.append(1 if label == "similar" else (-1 if label == "uncertain" else 0))
            pair_reports.append(f"[{log_label} -> {label}] {reasoning}")

    any_similar = any(label == "similar" for label in labels)
    any_uncertain = any(label == "uncertain" for label in labels)

    llm_result = 1 if any_similar else 0
    uncertain_flag = 1 if (any_uncertain and not any_similar) else 0
    reasoning_text = "\n\n".join(pair_reports)
    return llm_result, reasoning_text, uncertain_flag, pair_results
