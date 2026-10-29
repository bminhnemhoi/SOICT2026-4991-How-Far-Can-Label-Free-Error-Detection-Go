"""Faithful AnoLLM re-implementation per Tsai+ ICLR 2025.

Key design choices (from paper Sec 3 + Appendix Tab 7):
  - Backbone: SmolLM-135M (HuggingFaceTB/SmolLM-135M), pretrained init.
  - Serialization: "col_1: val_1, col_2: val_2, ..." per row.
  - Numerical binning: 10 equal-width buckets, label as digits 0-9.
  - Fine-tune: causal LM loss, AdamW lr=5e-5, 2000 steps, batch=8.
  - Random column permutation at training (each epoch) and at inference (r=21 perms).
  - Inference: anomaly score = mean negative log-likelihood across r permutations.

Setting: uncontaminated unsupervised — train on 50% of normals only,
test on remaining 50% normals + all anomalies.

Usage:
    from src.baselines.anollm_faithful import FaithfulAnoLLM
    model = FaithfulAnoLLM()
    model.fit(train_df, exclude_cols={"y_true", "row_id"})
    scores = model.score(test_df, exclude_cols={"y_true", "row_id"}, n_permutations=21)
"""
from __future__ import annotations
import math
import random
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM


def _quantize_numeric(series: pd.Series, n_buckets: int = 10) -> tuple[pd.Series, np.ndarray]:
    """Equal-width bin numeric column to digits 0-9 (string).
    Returns (binned_series_as_str, bin_edges).
    """
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return series.astype(str).fillna("?"), np.array([])
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        return pd.Series(["0"] * len(series), index=series.index), np.array([lo, hi])
    edges = np.linspace(lo, hi, n_buckets + 1)
    # Use np.digitize for binning (1-indexed, clip at 0..n_buckets-1)
    bins = np.clip(np.digitize(s.fillna(lo).values, edges) - 1, 0, n_buckets - 1)
    return pd.Series([str(int(b)) for b in bins], index=series.index), edges


def _serialize_row(row: dict, col_order: list[str]) -> str:
    """Render a row dict into AnoLLM-style text: col1: val1, col2: val2, ..."""
    parts = []
    for c in col_order:
        v = row.get(c, "")
        v = "?" if pd.isna(v) else str(v)
        parts.append(f"{c}: {v}")
    return ", ".join(parts)


class _TabularLMDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256, permute: bool = True):
        self.df = df.reset_index(drop=True)
        self.cols = list(df.columns)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.permute = permute

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx].to_dict()
        order = list(self.cols)
        if self.permute:
            random.shuffle(order)
        text = _serialize_row(row, order)
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


class FaithfulAnoLLM:
    def __init__(self,
                 model_name: str = "HuggingFaceTB/SmolLM-135M",
                 n_buckets: int = 10,
                 max_length: int = 256,
                 device: str | None = None,
                 seed: int = 42):
        self.model_name = model_name
        self.n_buckets = n_buckets
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.tokenizer = None
        self.model = None
        self.cols: list[str] = []
        self.numeric_cols: list[str] = []

    def _ensure_loaded(self):
        if self.tokenizer is None:
            # Prefer local snapshot if exists, else download
            import os
            local_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "SmolLM-135M")
            local_path = os.path.abspath(local_path)
            src = local_path if os.path.isdir(local_path) else self.model_name
            self.tokenizer = AutoTokenizer.from_pretrained(src)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                src, dtype=torch.float16, low_cpu_mem_usage=True)
            self.model.to(self.device)

    def _preprocess(self, df: pd.DataFrame, exclude_cols: set[str]) -> pd.DataFrame:
        """Drop excluded cols, bin numeric, return string-only DataFrame."""
        keep = [c for c in df.columns if c not in exclude_cols]
        out = pd.DataFrame(index=df.index)
        for c in keep:
            ser = df[c]
            num = pd.to_numeric(ser, errors="coerce")
            if num.notna().sum() / max(len(ser), 1) > 0.5:
                # Treat as numeric
                self.numeric_cols.append(c)
                binned, _ = _quantize_numeric(ser, self.n_buckets)
                out[c] = binned
            else:
                out[c] = ser.astype(str).fillna("?")
        return out

    def fit(self,
            train_df: pd.DataFrame,
            exclude_cols: Iterable[str] = ("y_true", "err_type", "row_id"),
            n_steps: int = 2000,
            batch_size: int = 1,
            lr: float = 5e-5,
            grad_accum: int = 4):
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        self.numeric_cols = []
        self._ensure_loaded()
        # Convert model to fp32 for training stability (fp16 AdamW causes OOM/unstable)
        self.model = self.model.float()
        # Enable gradient checkpointing to save memory at small speed cost
        self.model.gradient_checkpointing_enable()
        train = self._preprocess(train_df, set(exclude_cols))
        self.cols = list(train.columns)
        ds = _TabularLMDataset(train, self.tokenizer, self.max_length, permute=True)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.model.train()
        step = 0
        running_loss = 0.0
        opt.zero_grad()
        accum_count = 0
        while step < n_steps:
            for batch in loader:
                if step >= n_steps:
                    break
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out = self.model(**batch)
                loss = out.loss / grad_accum
                loss.backward()
                accum_count += 1
                if accum_count >= grad_accum:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad()
                    accum_count = 0
                running_loss += float(out.loss.item())
                step += 1
                if step % 200 == 0:
                    print(f"  [step {step}/{n_steps}] loss={running_loss/200:.4f}")
                    running_loss = 0.0
        if accum_count > 0:
            opt.step()
            opt.zero_grad()
        # Free optimizer state before inference
        del opt
        self.model.gradient_checkpointing_disable()
        self.model.eval()
        torch.cuda.empty_cache()
        # Keep fp32 for inference (more reliable than fp16 on some GPUs)
        print(f"  Training complete ({step} steps)")
        return self

    @torch.no_grad()
    def _row_nll(self, row_dict: dict, col_order: list[str]) -> float:
        """Compute per-token NLL for one row + column order."""
        text = _serialize_row(row_dict, col_order)
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             return_tensors="pt").to(self.device)
        ids = enc["input_ids"]
        if ids.shape[1] < 2:
            return 0.0
        out = self.model(input_ids=ids, labels=ids)
        # loss is mean per-token NLL of the sequence
        return float(out.loss.item())

    def score(self,
              test_df: pd.DataFrame,
              exclude_cols: Iterable[str] = ("y_true", "err_type", "row_id"),
              n_permutations: int = 21) -> np.ndarray:
        """Return per-row anomaly score = mean NLL across n permutations.
        Higher = more anomalous.
        """
        self._ensure_loaded()
        test = self._preprocess(test_df, set(exclude_cols))
        cols = list(test.columns)
        n = len(test)
        rng = random.Random(self.seed)
        # Pre-generate permutations (same for all rows for consistency)
        perms = [cols[:]]
        for _ in range(n_permutations - 1):
            p = cols[:]
            rng.shuffle(p)
            perms.append(p)
        scores = np.zeros(n, dtype=np.float64)
        self.model.eval()
        for i in range(n):
            if i % 50 == 0 and i > 0:
                print(f"  [score {i}/{n}]")
            row_dict = test.iloc[i].to_dict()
            nlls = [self._row_nll(row_dict, p) for p in perms]
            scores[i] = float(np.mean(nlls))
        return scores
