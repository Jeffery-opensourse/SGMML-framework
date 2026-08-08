"""Layer 1 - Text encoder (Qwen3.5-0.8B + LoRA).

Unstructured process descriptions (fiber type, preform structure,
manufacturing route, defect type) are encoded by a pretrained Qwen3.5-0.8B
language model adapted with Low-Rank Adaptation (LoRA) on the attention
weight matrices. The hidden states of the final transformer layer are
mean-pooled over the attention mask to obtain the text embedding.
"""
from __future__ import annotations

import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import (
    BASE_MODEL_ID,
    LORA_ALPHA,
    LORA_BATCH_SIZE,
    LORA_DROPOUT,
    LORA_EPOCHS,
    LORA_LEARNING_RATE,
    LORA_PATIENCE,
    LORA_R,
    LORA_SEED,
    LORA_TARGET_MODULES,
    LORA_VAL_RATIO,
    LORA_WEIGHT_DECAY,
    MAX_TEXT_LENGTH,
)


def set_seed(seed: int = LORA_SEED) -> None:
    """Fix all random sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextRegressionDataset(Dataset):
    """Pairs of tokenized text sequences and scalar targets."""

    def __init__(self, texts: List[str], targets: np.ndarray,
                 tokenizer, max_len: int = MAX_TEXT_LENGTH):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int):
        enc = self.tokenizer(
            self.texts[i],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "target": torch.tensor(self.targets[i], dtype=torch.float32),
        }


class QwenRegressor(nn.Module):
    """LoRA-adapted Qwen backbone with a small regression head.

    ``encode`` returns the attention-masked mean pooling of the last hidden
    state, which is the text embedding used in Layer 2.
    """

    def __init__(self, qwen_lora, hidden_size: int):
        super().__init__()
        self.backbone = qwen_lora
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Dropout(LORA_DROPOUT),
            nn.Linear(128, 1),
        )
        self.hidden_size = hidden_size

    def encode(self, input_ids, attention_mask) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask, target=None):
        pooled = self.encode(input_ids, attention_mask)
        pred = self.reg_head(pooled).squeeze(-1)
        if target is not None:
            return F.mse_loss(pred, target), pred, pooled
        return pred, pooled


def load_base_model(model_id: str = BASE_MODEL_ID, cache_dir: Optional[str] = None,
                    device: Optional[torch.device] = None):
    """Load the Qwen base model, tokenizer and device."""
    device = device or get_device()
    model_dir = model_id
    if cache_dir:
        model_dir = os.path.join(cache_dir, model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    return tokenizer, model, device


def train_lora(texts_train: List[str], y_train: np.ndarray,
               model_id: str = BASE_MODEL_ID, cache_dir: Optional[str] = None,
               seed: int = LORA_SEED, verbose: bool = True):
    """Fine-tune the Qwen text encoder with LoRA.

    A small validation split inside the training texts monitors the loss
    with an early-stopping criterion. Returns the trained regressor, the
    tokenizer, the device and the best validation loss.
    """
    set_seed(seed)
    idx = np.arange(len(texts_train))
    tr_idx, va_idx = train_test_split(
        idx, test_size=LORA_VAL_RATIO, random_state=LORA_SEED)

    texts_tr = [texts_train[i] for i in tr_idx]
    y_tr = y_train[tr_idx]
    texts_va = [texts_train[i] for i in va_idx]
    y_va = y_train[va_idx]

    tokenizer, base_model, device = load_base_model(model_id, cache_dir)
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    qwen_lora = get_peft_model(base_model, lora_config)
    hidden_size = base_model.config.hidden_size
    regressor = QwenRegressor(qwen_lora, hidden_size).to(device)

    loader_tr = DataLoader(
        TextRegressionDataset(texts_tr, y_tr, tokenizer),
        batch_size=LORA_BATCH_SIZE, shuffle=True,
    )
    loader_va = DataLoader(
        TextRegressionDataset(texts_va, y_va, tokenizer),
        batch_size=LORA_BATCH_SIZE, shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        [p for p in regressor.parameters() if p.requires_grad],
        lr=LORA_LEARNING_RATE,
        weight_decay=LORA_WEIGHT_DECAY,
    )

    best_val, best_state, patience = float("inf"), None, 0
    for epoch in range(LORA_EPOCHS):
        regressor.train()
        total_loss = 0.0
        for batch in loader_tr:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["target"].to(device)
            loss, _, _ = regressor(input_ids, attention_mask, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in regressor.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_train = total_loss / len(loader_tr)
        avg_val = eval_loss(regressor, loader_va, device)
        if verbose:
            print(f"    E{epoch + 1}: train={avg_train:.4f}, val={avg_val:.4f}")
        if avg_val < best_val - 1e-4:
            best_val = avg_val
            best_state = {k: v.detach().cpu().clone()
                          for k, v in regressor.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= LORA_PATIENCE:
                if verbose:
                    print(f"    early stop at epoch {epoch + 1}")
                break

    if best_state is not None:
        regressor.load_state_dict(best_state)
    return regressor, tokenizer, device, best_val


def eval_loss(regressor: QwenRegressor, loader: DataLoader,
              device: torch.device) -> float:
    """Mean squared loss on a validation loader."""
    regressor.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["target"].to(device)
            loss, _, _ = regressor(input_ids, attention_mask, target)
            losses.append(loss.item())
    regressor.train()
    return float(np.mean(losses))


def extract_embeddings(regressor: QwenRegressor, tokenizer,
                       texts: List[str], device: torch.device,
                       max_len: int = MAX_TEXT_LENGTH) -> np.ndarray:
    """Encode texts into the Layer 1 embedding matrix (float32)."""
    regressor.eval()
    embeddings = []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, padding=True, truncation=True,
                            max_length=max_len, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            pooled = regressor.encode(enc["input_ids"], enc["attention_mask"])
            embeddings.append(pooled.squeeze(0).float().cpu().numpy())
    return np.array(embeddings, dtype=np.float32)


def load_text_encoder(model_id: str = BASE_MODEL_ID, lora_dir: str = "",
                      cache_dir: Optional[str] = None,
                      device: Optional[torch.device] = None):
    """Load a saved LoRA regressor for inference.

    ``lora_dir`` points to the folder containing the saved LoRA adapter
    (e.g. ``weights/fusion_model/qwen_lora_reg``).
    """
    device = device or get_device()
    tokenizer, base_model, device = load_base_model(model_id, cache_dir, device)
    qwen_lora = PeftModel.from_pretrained(base_model, lora_dir)
    hidden_size = base_model.config.hidden_size
    regressor = QwenRegressor(qwen_lora, hidden_size).to(device)
    return regressor, tokenizer, device
