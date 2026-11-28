"""Métricas de recuperação, ranking e qualidade da lista."""
from __future__ import annotations

import numpy as np
import pandas as pd

K = 10
K_CAND = 100


def dcg(g) -> float:
    g = np.asarray(g, dtype=float)
    return float(((2 ** g - 1) / np.log2(np.arange(2, len(g) + 2))).sum())


def ndcg(lista: np.ndarray, grades: np.ndarray, k: int = K) -> float:
    ideal = dcg(np.sort(grades)[::-1][:k])
    return dcg(grades[lista[:k]]) / ideal if ideal > 0 else 0.0


def mrr(lista: np.ndarray, grades: np.ndarray, k: int = K) -> float:
    for i, j in enumerate(lista[:k]):
        if grades[j] == 2:
            return 1.0 / (i + 1)
    return 0.0


def metricas(lista: np.ndarray, grades: np.ndarray, df: pd.DataFrame) -> dict:
    top = lista[:K]
    rel = grades > 0
    return {
        "Compat@10": float(rel[top].mean()) if len(top) else 0.0,
        "Recall@100": float(rel[lista[:K_CAND]].sum() / rel.sum()),
        "nDCG@10": ndcg(lista, grades),
        "MRR": mrr(lista, grades),
        "Diversidade@10": df["marca"].iloc[top].nunique() / K if len(top) else 0.0,
        "Itens na lista": len(top),
    }


def bootstrap_dif(a: np.ndarray, b: np.ndarray, n: int = 5000, seed: int = 0):
    """Diferença média pareada (a - b) com IC 95% por bootstrap."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
