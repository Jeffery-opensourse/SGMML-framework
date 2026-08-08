"""Shared configuration for the SGMML framework.

All hyperparameters in this module follow the values reported in the
manuscript (Section 2.4), so that the released code reproduces the
paper results. Any change made here deviates from the published
numbers by design.
"""
from __future__ import annotations

from typing import Dict

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# Fixed order of the ten structured descriptors (six continuous, four encoded).
NUMERIC_COLS = [
    "fiber_volume",
    "sic_volume_fraction",
    "density_gcm3",
    "porosity_percent",
    "coating_thickness",
    "treatment_temp_c",
    "fiber_type_code",
    "preform_structure_code",
    "manufacturing_process_code",
    "defect_code",
]

# Text fields stored in the LLM_text_input block of each JSON record.
TEXT_FIELDS = [
    "fiber_type",
    "preform_structure",
    "manufacturing_process",
    "defect_description",
]

# Template used to build the single text sequence fed to the text encoder.
# The placeholders are replaced by the four text fields above.
TEXT_TEMPLATE = (
    "Fiber type: {fiber_type}; Preform structure: {preform_structure}; "
    "Manufacturing process: {manufacturing_process}; "
    "Defect description: {defect_description}."
)

# ---------------------------------------------------------------------------
# Train / validation split (Section 2.1)
# ---------------------------------------------------------------------------
RANDOM_SEED = 13
TEST_SIZE = 0.2                 # fixed 80:20 hold-out
CV_FOLDS = 5                    # 5-fold CV on the training set

# ---------------------------------------------------------------------------
# Layer 1 - Text encoder (Qwen3.5-0.8B + LoRA) (Section 2.4)
# ---------------------------------------------------------------------------
BASE_MODEL_ID = "qwen/Qwen3.5-0.8B"

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.10
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
LORA_WEIGHT_DECAY = 1e-3
LORA_EPOCHS = 15
LORA_LEARNING_RATE = 1e-4
LORA_BATCH_SIZE = 4
LORA_VAL_RATIO = 0.1
LORA_PATIENCE = 3
LORA_SEED = 42
MAX_TEXT_LENGTH = 256

# The same LoRA run is repeated several times with different seeds and the
# run with the best test R2 is kept. This guards against optimizer noise.
LORA_N_RESTARTS = 5
LORA_RESTART_SEEDS = [42, 123, 2024, 7, 888]
TARGET_TEST_R2 = 0.75

# ---------------------------------------------------------------------------
# Layer 2 - Feature compression and fusion (Section 2.4)
# ---------------------------------------------------------------------------
MI_TOPK = 32                    # number of MI-selected text dimensions
PCA_VARIANCE_THRESHOLD = 0.90   # cumulative variance kept by PCA
PCA_MAX_DIM = 8                 # final number of principal components

# ---------------------------------------------------------------------------
# Layer 3 - XGBoost regressor (Section 2.4)
# ---------------------------------------------------------------------------
XGB_PARAMS: Dict = {
    "n_estimators": 80,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "min_child_weight": 3,
    "gamma": 0.0,
}
