
from Agents.false_labelled_control_gate import compare_signature_sets
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import re
from typing import List, Optional, Tuple


def to_image_list(cell) -> List[np.ndarray]:
    """Normalize a dataframe cell into a flat list of 2-D (224, 224) image arrays.

    A cell may hold a single image, a stack of images, or a (possibly nested)
    list of images — the sign-detection columns can contain multiple detections.
    Nesting is flattened recursively down to the individual 2-D signature arrays.
    """
    if cell is None:
        return []
    if isinstance(cell, np.ndarray):
        if cell.dtype == object:                      # object array of sub-arrays
            images: List[np.ndarray] = []
            for item in cell:
                images.extend(to_image_list(item))
            return images
        if cell.ndim == 2:                            # one (224, 224) image
            return [cell]
        if cell.ndim == 3:                            # (N, 224, 224) stack
            return [cell[i] for i in range(cell.shape[0])]
        if cell.ndim >= 4:                            # e.g. (N, 224, 224, 1)
            return [np.squeeze(cell[i]) for i in range(cell.shape[0])]
        return []                                     # 0-d / 1-d: nothing usable
    if isinstance(cell, (list, tuple)):
        images = []
        for item in cell:
            images.extend(to_image_list(item))
        return images
    return to_image_list(np.asarray(cell))
 
def extract_verdict(text: str) -> Optional[Tuple[str, str]]:
    """
    Robustly extracts verdict and detailed_reasoning from LLM output.
    Does NOT rely on JSON parsing.
    Designed for free-form LLM reasoning.
    """
 
    if not isinstance(text, str) or not text.strip():
        return None
 
    # 1️⃣ Focus only on the Verdict section
    verdict_section = re.search(
        r"\*\*4\)\s*Verdict:\*\*(.*)$",
        text,
        re.DOTALL | re.IGNORECASE
    )
 
    if not verdict_section:
        return None
 
    section = verdict_section.group(1)
 
    # 2️⃣ Extract verdict (semantic, not syntactic)
    verdict_match = re.search(
        r"\b(similiar|similar|dissimiliar|dissimilar)\b",
        section,
        re.IGNORECASE
    )
 
    if not verdict_match:
        return None
 
    verdict = verdict_match.group(1).lower()
    verdict = "similar" if "similar" in verdict else "dissimilar"
 
    # 3️⃣ Extract reasoning (everything except verdict line)
    reasoning_match = re.search(
        r"detailed_reasoning[^:]*[:\-]?\s*(.*)",
        section,
        re.DOTALL | re.IGNORECASE
    )
 
    if reasoning_match:
        reason = reasoning_match.group(1).strip()
    else:
        # Fallback: use whole section without verdict word
        reason = re.sub(
            r"\b(similiar|similar|dissimiliar|dissimilar)\b",
            "",
            section,
            flags=re.IGNORECASE
        ).strip()
 
    return verdict, reason
 
# Default column names. Each detection column holds (possibly multiple)
# preprocessed (224, 224) signature arrays (float, normalized to [0, 1]).
BBHS_COL = "bbhs_sign_detections"            # column with the bbhs signature detections
TALIMAT_COL = "talimat_sign_detections"      # column with the talimat signature detections


def run_signature_evaluation(
    df,
    bbhs_col: str = BBHS_COL,
    talimat_col: str = TALIMAT_COL,
):
    """Run the LLM signature gate over an in-memory dataframe and return a new one.

    For each row, the detections in both columns are flattened and every
    bbhs × talimat pair is sent to the model (2 images per call). Results are
    written into new columns on a copy of the input (the caller's df is not
    mutated):

        llm_result       : 1 = similar, 0 = dissimilar (<NA> if not comparable)
        llm_pair_results : per-pair list ordered bbhs-outer/talimat-inner,
                           encoded 1=similar, 0=dissimilar, -1=uncertain
                           e.g. [0, 0, 1, 0, -1, 0]
        llm_reasoning    : the model's reasoning (per-pair, aggregated)
        llm_uncertain    : flag -> 1 = at least one pair was uncertain and none similar

    Usage (e.g. from the Jupyter notebook that already holds ``df``):
        output_df = run_signature_evaluation(df)
    """
    output_df = df.copy()

    llm_results = []
    llm_reasonings = []
    llm_uncertain = []
    llm_pair_results = []
    for idx, row in output_df.iterrows():
        bbhs_images = to_image_list(row[bbhs_col])
        talimat_images = to_image_list(row[talimat_col])

        result, reasoning, uncertain, pair_results = compare_signature_sets(
            bbhs_images, talimat_images
        )
        llm_results.append(result)
        llm_reasonings.append(reasoning)
        llm_uncertain.append(uncertain)
        llm_pair_results.append(pair_results)

    output_df["llm_result"] = pd.array(llm_results, dtype="Int64")  # nullable int keeps <NA>
    output_df["llm_pair_results"] = llm_pair_results
    output_df["llm_reasoning"] = llm_reasonings
    output_df["llm_uncertain"] = llm_uncertain
    return output_df
