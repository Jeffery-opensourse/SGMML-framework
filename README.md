# SGMML: Semantics-Guided Multimodal Machine Learning for C/C-SiC Flexural Strength Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of the SGMML framework for predicting the flexural
strength of **C/C-SiC composites** used in extreme environments, described in:

> *A Semantics-Guided Multimodal Machine Learning Framework for Predicting the Flexural Strength of C/C-SiC Composites*

## Overview

C/C-SiC composites are critical thermostructural materials whose flexural
strength is governed by complex nonlinear process-structure-property
relationships. Relying only on hand-crafted numerical descriptors fails to
capture the microstructural semantics embedded in manufacturing process
texts, while pure LLMs are imprecise for numerical regression on small
datasets.

SGMML combines the strengths of both through a three-layer architecture:

1. **Layer 1 (Textual encoding).** A LoRA-fine-tuned Qwen3.5-0.8B encodes the
   manufacturing process text (fiber type, preform structure, manufacturing
   route, defect description) into semantic embeddings via attention-masked
   mean pooling of the last hidden state.
2. **Layer 2 (Feature compression & fusion).** The high-dimensional text
   embeddings are reduced by mutual-information (MI) selection and PCA under
   a 90% cumulative explained-variance threshold, then early-concatenated
   with ten standardized numerical descriptors.
3. **Layer 3 (Regression prediction).** The fused feature vector is fed into
   an XGBoost regressor (optionally combined into an XGB+RF+Ridge stacking
   ensemble).

With 142 literature-derived samples, the model reaches an **MAE of 25.96 MPa
and R² of 0.81** on an independent 20% hold-out test set, outperforming
traditional ML models by 20.9% and fine-tuned LLMs by 92.9% in terms of R².

## Repository layout

```
sgmml_opensource/
├── sgmml/                  # Core package
│   ├── config.py           # Shared hyperparameters (Section 2.4)
│   ├── data.py             # Data loading & preprocessing (Section 2.1)
│   ├── metrics.py          # Evaluation metrics (Section 2.5)
│   ├── ml_models.py        # Eight conventional ML baselines (Section 2.3.2)
│   ├── llm_api.py          # Closed-source LLM zero/few-shot API (Section 2.3.3)
│   ├── text_encoder.py     # Layer 1: Qwen + LoRA text encoder
│   ├── fusion.py           # Layer 2: MI selection + PCA compression
│   ├── regressor.py        # Layer 3: XGBoost regression + stacking
│   └── trainer.py          # End-to-end training (main_train) & inference (Predictor)
├── scripts/                # Command-line entry points
│   ├── train.py
│   ├── predict.py
│   ├── predict_api.py
│   └── benchmark_ml.py
├── examples/               # Usage examples
├── data/                   # New-sample templates
├── requirements.txt
└── README.md
```

## Getting started

### Installation

```bash
pip install -r requirements.txt
```

### Data format

Each dataset is a JSON list of records, one per sample:

```json
{
  "sample_id": "sample_1",
  "ML_input": {
    "fiber_volume": null,
    "sic_volume_fraction": null,
    "density_gcm3": null,
    "porosity_percent": null,
    "coating_thickness": 0.0,
    "treatment_temp_c": 1600.0,
    "fiber_type_code": 3,
    "preform_structure_code": 6,
    "manufacturing_process_code": 1,
    "defect_code": 6
  },
  "LLM_text_input": {
    "fiber_type": "PAN-based carbon fibers (T300, 12K, Toray, Japan)",
    "preform_structure": "3D needle-punched carbon preform: 0 deg non-woven fiber cloths, 90 deg non-woven fiber cloths and short-cut fiber web layers overlapped by turns and needled to construct fiber felts",
    "manufacturing_process": "PIP (PSO precursor pyrolysis at 1400 C) + CVI (PyC, 900-1200 C) + LSI (1500-1700 C)",
    "defect_description": "macro pores around the needle-punched fiber bundles are not completely filled with molten Si; the generated SiC is not dense and contains micro pores in the interior of SiC"
  },
  "target": {"flexural_strength_MPa": 212.0}
}
```

* Ten `ML_input` descriptors: six continuous (fiber volume fraction, SiC
  volume fraction, density, porosity, coating thickness, treatment
  temperature) and four label-encoded categorical variables (fiber type,
  preform structure, manufacturing process, defect category).
* The four `LLM_text_input` fields form the text sequence fed to Layer 1.
* Missing numerical values are imputed with the column mode; categorical
  variables use label encoding. For prediction-only records set
  `target.flexural_strength_MPa` to `null`.
* The bundled `data/example_samples.json` contains two records taken from
  Ma et al., Ceram. Int. 2021, 47, 24130-24138 (Table 2) and is intended to
  illustrate the data format; the full training dataset is released
  separately. See also the "External validation" section below.

### Training

```bash
python scripts/train.py --data path/to/dataset.json --save_dir output/model
```

The script follows the paper's protocol: a fixed 80:20 hold-out split
(`random_state=13`), LoRA fine-tuning of the Qwen text encoder (with several
restarts, keeping the best test R²), MI + PCA compression fitted on the
training set only, and an XGBoost regressor on the fused features. It also
reports 5-fold cross-validation and segmented (low/mid/high strength) errors.

### Prediction on new samples

```bash
python scripts/predict.py --model_dir output/model \
    --data path/to/new_samples.json
```

`--model_dir` must contain the artifacts produced by `train.py`: `num_scaler.pkl`,
`mi_topk_idx.pkl`, `text_pca.pkl`, `xgb_model.pkl`, `rf_model.pkl`,
`ridge_model.pkl`, `meta_ridge.pkl`, `reg_head.pt`, `hidden_size.json` and the
`qwen_lora_reg/` LoRA adapter.

### Closed-source LLM baselines

```bash
export OPENAI_API_KEY=...        # OpenAI-compatible endpoint
export DASHSCOPE_API_KEY=...     # Alibaba Cloud DashScope (Qwen-Plus / DeepSeek)
python scripts/predict_api.py --data path/to/new_samples.json --mode few_shot
```

All API calls use `temperature=0.0` for deterministic output.

### Conventional ML benchmark

```bash
python scripts/benchmark_ml.py --data path/to/dataset.json
```

## Reproducibility

- Fixed train/test split: `random_state=13`, 80:20 hold-out
- LoRA internal validation split: `random_state=42`
- MI selection: top-32 text dimensions; PCA keeps ~90% cumulative variance
  (8 principal components, `feature_dim = 10 + 8 = 18`)
- XGBoost: 80 trees, `max_depth=3`, `learning_rate=0.05`
- Metrics: R², MSE, RMSE, MAE, MAPE

All hyperparameters live in `sgmml/config.py` and match the values reported
in the manuscript, so the released code reproduces the paper results.

## External validation

The independent external validation samples (8 literature cases) and the two
experimental samples used in the manuscript are released alongside the code.
See the manuscript Section 3.3 and Table S9 for the corresponding results.
The bundled `data/example_samples.json` reproduces two records from Ma et al.
(2021) so that the input format can be inspected without the full dataset.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{wang2026sgmml,
  title={A Semantics-Guided Multimodal Machine Learning Framework for Predicting
         the Flexural Strength of C/C-SiC Composites},
  author={Wang, C. and others},
  journal={Extreme Materials},
  year={2026}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file
for details.
