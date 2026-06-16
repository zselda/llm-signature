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
 
def signature_classifier(images, pagination_mode=True):
    """Classify a signature pair as similar / dissimilar.

    ``images`` is a sequence of NumPy arrays (one (224, 224) array per signature).
    With ``pagination_mode=False`` (the false-labelled flow) the whole pair is sent
    to the model in a single multimodal request; with ``pagination_mode=True`` each
    array is sent individually.
    """
    start_time2 = datetime.now()

    SYSTEM_PROMPT = prompt_lib.agent_signature_false_evaluator


    USER_PROMPT = """ """
    results = {}
    if pagination_mode:
        for idx, array in enumerate(images):
            start_time = datetime.now()
            print("processing image", idx)
            result = client.generate_with_image(
                model="gemma_27b",
                images=[array],
                system_prompt=SYSTEM_PROMPT,
                user_prompt=USER_PROMPT
            )

            end_time = datetime.now()

            print(end_time - start_time)

            print(result)
            results[idx] = result
        end_time2 = datetime.now()
        print(end_time2 - start_time2)

    else:
        start_time = datetime.now()
        print("processing pair")
        results = client.generate_with_image(
            model="gemma_27b",
            images=images,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=USER_PROMPT
        )

        end_time = datetime.now()

        print(end_time - start_time)

        print(results)

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


def compare_signature_sets(bbhs_images, talimat_images):
    """Compare every bbhs × talimat detection pair and aggregate to a row verdict.

    Each pair is sent to the model as exactly two images (matching the prompt's
    "2 signatures" framing). Aggregation:
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

    labels = []
    pair_results = []
    pair_reports = []
    for bi, bbhs_img in enumerate(bbhs_images):
        for ti, talimat_img in enumerate(talimat_images):
            raw = signature_classifier([bbhs_img, talimat_img], pagination_mode=False)
            label, reasoning = classify_verdict(raw)
            labels.append(label)
            # 1 = similar, 0 = dissimilar, -1 = uncertain
            pair_results.append(1 if label == "similar" else (-1 if label == "uncertain" else 0))
            pair_reports.append(f"[bbhs#{bi} vs talimat#{ti} -> {label}] {reasoning}")

    any_similar = any(label == "similar" for label in labels)
    any_uncertain = any(label == "uncertain" for label in labels)

    llm_result = 1 if any_similar else 0
    uncertain_flag = 1 if (any_uncertain and not any_similar) else 0
    reasoning_text = "\n\n".join(pair_reports)
    return llm_result, reasoning_text, uncertain_flag, pair_results
