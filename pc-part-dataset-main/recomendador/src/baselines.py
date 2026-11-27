"""Alternativas simples usadas como comparação (baselines)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

_TOK = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokenizar(t: str) -> list[str]:
    return _TOK.findall(t.lower())


class Lexical:
    """Palavra-chave (conta termos em comum) e BM25 sobre os textos de cada categoria."""

    def __init__(self, textos_por_cat: dict[str, list[str]]):
        self.toks = {c: [tokenizar(t) for t in ts] for c, ts in textos_por_cat.items()}
        self.sets = {c: [set(t) for t in ts] for c, ts in self.toks.items()}
        self.bm25 = {c: BM25Okapi(ts) for c, ts in self.toks.items()}

    def palavra_chave(self, cat: str, consulta: str) -> np.ndarray:
        q = set(tokenizar(consulta))
        return np.array([len(q & s) for s in self.sets[cat]], dtype=float)

    def bm25_scores(self, cat: str, consulta: str) -> np.ndarray:
        return self.bm25[cat].get_scores(tokenizar(consulta))


def popularidade(df: pd.DataFrame) -> np.ndarray:
    return -df["pop_rank"].to_numpy()


def ordenar_por_score(score: np.ndarray, desempate: np.ndarray, k: int,
                      valido: np.ndarray | None = None) -> np.ndarray:
    s = score.astype(float).copy()
    if valido is not None:
        s[~valido] = -np.inf
    ordem = np.lexsort((desempate, -s))  # maior score; empate -> mais popular
    ordem = ordem[np.isfinite(s[ordem])]
    return ordem[:k]
