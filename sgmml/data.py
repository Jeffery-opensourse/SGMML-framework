"""Data loading and preprocessing (Section 2.1).

Every record in the dataset JSON contains a structured ``ML_input`` block,
a free-text ``LLM_text_input`` block and an optional ``target`` block. This
module turns the raw JSON into a DataFrame, imputes missing numerical values
with the mode, exposes the categorical encoding map and builds the combined
text sequence used by the text encoder.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import NUMERIC_COLS, TEXT_FIELDS, TEXT_TEMPLATE

# Column names of the four text fields once flattened into a DataFrame.
TEXT_COLUMNS = [
    "fiber_type_text",
    "preform_structure_text",
    "manufacturing_process_text",
    "defect_description_text",
]


def load_dataset(path: str | Path) -> List[Dict[str, Any]]:
    """Load a dataset stored as a JSON list of sample records."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_records(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten raw JSON records into a structured DataFrame.

    Numerical values are taken from ``ML_input``, the four free-text fields
    from ``LLM_text_input`` and the regression target from ``target``.
    """
    records = []
    for item in data:
        ml = item["ML_input"]
        text = item["LLM_text_input"]
        target = item.get("target", {})
        records.append(
            {
                "sample_id": item.get("sample_id", ""),
                **{col: ml[col] for col in NUMERIC_COLS},
                "fiber_type_text": text[TEXT_FIELDS[0]],
                "preform_structure_text": text[TEXT_FIELDS[1]],
                "manufacturing_process_text": text[TEXT_FIELDS[2]],
                "defect_description_text": text[TEXT_FIELDS[3]],
                "flexural_strength": target.get("flexural_strength_MPa", np.nan),
            }
        )
    return pd.DataFrame(records)


def build_text(row: pd.Series) -> str:
    """Assemble the combined text sequence for a single sample row."""
    return TEXT_TEMPLATE.format(
        fiber_type=row["fiber_type_text"],
        preform_structure=row["preform_structure_text"],
        manufacturing_process=row["manufacturing_process_text"],
        defect_description=row["defect_description_text"],
    )


def build_texts(df: pd.DataFrame) -> List[str]:
    """Build the combined text sequence for every row of a DataFrame."""
    return [build_text(row) for _, row in df.iterrows()]


def build_numeric_matrix(
    df: pd.DataFrame, numeric_cols: Optional[List[str]] = None
) -> np.ndarray:
    """Build the raw numerical feature matrix as float32."""
    cols = numeric_cols or NUMERIC_COLS
    return df[cols].values.astype(np.float32)


def build_llm_prompt(sample: Dict[str, Any]) -> str:
    """Build the prompt passed to closed-source LLMs for a single sample.

    The prompt mirrors the template used in the manuscript (Fig. S2): it
    lists the four text descriptors and the six continuous numerical
    features, then asks for a single numerical answer in MPa.
    """
    ml = sample["ML_input"]
    text = sample["LLM_text_input"]
    numeric = ", ".join(
        f"{col}: {ml[col]}" for col in NUMERIC_COLS[:6]
    )
    return (
        f"Predict the flexural strength (MPa) of the C/C-SiC composite: "
        f"fiber type: {text[TEXT_FIELDS[0]]}, "
        f"preform structure: {text[TEXT_FIELDS[1]]}, "
        f"manufacturing process: {text[TEXT_FIELDS[2]]}, "
        f"defect type: {text[TEXT_FIELDS[3]]}, "
        f"{numeric}. Answer:"
    )


def impute_with_mode(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in numerical columns with the column mode.

    The manuscript reports that missing numerical descriptors were imputed
    with the mode of the training set before model development. Columns that
    are entirely missing (no observed value to derive a mode from) are left
    untouched.
    """
    out = df.copy()
    for col in NUMERIC_COLS:
        if out[col].isna().any():
            mode_vals = out[col].mode()
            if not mode_vals.empty:
                out[col] = out[col].fillna(mode_vals.iloc[0])
    return out


def encoding_table(df: pd.DataFrame) -> Dict[str, Dict[int, str]]:
    """Map each categorical code to the text descriptions seen in the data.

    Returns ``{code_column: {code: "text / text / ..."}}``.
    """
    pairs = [
        ("fiber_type_code", "fiber_type_text"),
        ("preform_structure_code", "preform_structure_text"),
        ("manufacturing_process_code", "manufacturing_process_text"),
        ("defect_code", "defect_description_text"),
    ]
    maps: Dict[str, Dict[int, str]] = {}
    for code_col, text_col in pairs:
        grouped = (
            df.groupby(code_col)[text_col]
            .unique()
            .apply(lambda values: " / ".join(str(v) for v in values))
            .to_dict()
        )
        maps[code_col] = {int(k): v for k, v in grouped.items()}
    return maps


def missing_value_report(
    df: pd.DataFrame, numeric_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Report the number and proportion of missing values per feature."""
    cols = numeric_cols or NUMERIC_COLS
    total = len(df)
    rows = []
    for col in cols:
        n_missing = int(df[col].isna().sum())
        rows.append(
            {
                "feature": col,
                "n_missing": n_missing,
                "missing_pct": round(n_missing / total * 100, 2),
            }
        )
    return pd.DataFrame(rows)
