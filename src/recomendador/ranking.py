"""Etapa 2 — ranqueamento aprendido (LightGBM LambdaMART) + diversidade (MMR).

As variáveis são RELACIONAIS (comparam o item com a montagem: sobra de potência,
folga de slots, diferença de faixa de preço…), então valem para qualquer socket,
inclusive os que o modelo nunca viu.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .dados import RAIZ, potencia_recomendada

ARQ_RANKER = RAIZ / "modelos" / "ranker_lambdamart.txt"
ALVO_ID = {"motherboard": 0, "memory": 1, "case": 2, "power-supply": 3,
           "video-card": 4, "cpu-cooler": 5, "internal-hard-drive": 6, "cpu": 7}
_EFIC = {"plus": 0, "bronze": 1, "silver": 2, "gold": 3, "platinum": 4, "titanium": 5}

# Todas as variáveis são RELATIVAS à consulta (percentis entre os candidatos, faixa da CPU
# dentro da própria plataforma, folgas em relação à montagem). Valores absolutos (preço,
# MHz, popularidade global) mudam de uma geração para outra e não generalizam.
GRUPOS = {
    "neural": ["sim_z", "rank_denso", "rrf"],
    "lexical": ["bm25_rel", "rank_bm25"],
    "preco": ["preco_pct_q", "dif_faixa", "outlier"],
    "popularidade": ["pop_q"],
    "relacionais": ["folga_slots", "uso_memoria", "gb_pct_q", "speed_pct_q", "tam_placa",
                    "folga_gabinete", "sobra_potencia", "eficiencia"],
    "categoria": ["alvo_id"],
}
FEATURES = [f for g in GRUPOS.values() for f in g]


def _g(s: pd.Series | None, col: str) -> float:
    if s is None:
        return np.nan
    v = s.get(col)
    return float(v) if v is not None and pd.notna(v) else np.nan


def _pct(v) -> np.ndarray:
    """Percentil de cada valor dentro da lista de candidatos da consulta."""
    return pd.Series(np.asarray(v, dtype=float)).rank(pct=True).to_numpy()


def montar_features(alvo: str, df: pd.DataFrame, cand: np.ndarray, montagem: dict,
                    s_denso: np.ndarray, s_bm25: np.ndarray, r_denso: np.ndarray,
                    r_bm25: np.ndarray, f_rrf: np.ndarray) -> pd.DataFrame:
    sub = df.iloc[cand]
    n = len(cand)
    cpu = montagem.get("cpu")
    mb = montagem.get("motherboard")
    gpu = montagem.get("video-card")
    sim = s_denso[cand]
    bm = s_bm25[cand]
    preco_q = _pct(sub["price"])
    X = {
        "sim_z": (sim - sim.mean()) / (sim.std() + 1e-9),
        "rank_denso": np.where(np.isfinite(r_denso[cand]), r_denso[cand], 999),
        "rrf": f_rrf[cand],
        "bm25_rel": bm / (bm.max() + 1e-9),
        "rank_bm25": np.where(np.isfinite(r_bm25[cand]), r_bm25[cand], 999),
        "preco_pct_q": preco_q,
        "dif_faixa": np.abs(preco_q - _g(cpu, "pct_plataforma")),
        "outlier": sub["outlier_preco"].to_numpy().astype(float),
        "pop_q": _pct(1.0 - sub["pop_rank"].to_numpy()),
        "alvo_id": np.full(n, ALVO_ID.get(alvo, -1)),
    }
    nan = np.full(n, np.nan)
    for c in GRUPOS["relacionais"]:
        X[c] = nan.copy()
    if alvo == "memory":
        X["gb_pct_q"] = _pct(sub["total_gb"])
        X["speed_pct_q"] = _pct(sub["speed_mhz"])
        X["folga_slots"] = _g(mb, "memory_slots") - sub["n_modules"].to_numpy()
        X["uso_memoria"] = sub["total_gb"].to_numpy() / _g(mb, "max_memory")
    elif alvo == "motherboard":
        X["tam_placa"] = sub["tam_placa"].to_numpy()
        X["gb_pct_q"] = _pct(sub["max_memory"])
    elif alvo == "case":
        X["folga_gabinete"] = sub["tam_max"].to_numpy() - _g(mb, "tam_placa")
    elif alvo == "power-supply":
        req = potencia_recomendada(_g(cpu, "tdp"), _g(gpu, "watts") if gpu is not None else 0.0)
        X["sobra_potencia"] = sub["wattage"].to_numpy() / req - 1
        X["eficiencia"] = sub["efficiency"].map(_EFIC).to_numpy(dtype=float)
    return pd.DataFrame(X)[FEATURES]


# ---------------------------------------------------------------------------
# Treino / carga do LambdaMART
# ---------------------------------------------------------------------------
HIPER = {"leve": {"n_estimators": 100, "num_leaves": 7},
         "medio": {"n_estimators": 300, "num_leaves": 15}}


def treinar(X: pd.DataFrame, y: np.ndarray, grupos: list[int], features: list[str] | None = None,
            hiper: str = "medio"):
    import lightgbm as lgb
    features = features or FEATURES
    m = lgb.LGBMRanker(objective="lambdarank", learning_rate=0.05, min_child_samples=20,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                       random_state=0, verbose=-1, **HIPER[hiper])
    m.fit(X[features], y, group=grupos)
    m.features_ = features
    return m


def salvar(m, arq=ARQ_RANKER):
    arq.parent.mkdir(exist_ok=True)
    # a biblioteca C do LightGBM não lida com acentos no caminho (Windows): grava via Python
    arq.write_text(m.booster_.model_to_string(), encoding="utf-8")
    arq.with_suffix(".json").write_text(json.dumps({"features": m.features_}, indent=2))


class RankerCarregado:
    def __init__(self, arq=ARQ_RANKER):
        import lightgbm as lgb
        self.booster = lgb.Booster(model_str=arq.read_text(encoding="utf-8"))
        self.features_ = json.loads(arq.with_suffix(".json").read_text())["features"]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(X[self.features_])


def carregar(arq=ARQ_RANKER):
    return RankerCarregado(arq) if arq.exists() else None


# ---------------------------------------------------------------------------
# Diversidade
# ---------------------------------------------------------------------------
def mmr_marcas(scores: np.ndarray, marcas: np.ndarray, n: int, lam: float) -> np.ndarray:
    """Reordena penalizando marcas repetidas (scores normalizados por z-score)."""
    if len(scores) == 0:
        return np.array([], dtype=int)
    s = (scores - scores.mean()) / (scores.std() + 1e-9)
    restantes = list(np.argsort(-s)[: max(5 * n, 50)])
    escolhidos, cont = [], {}
    while restantes and len(escolhidos) < n:
        aj = [s[i] - lam * cont.get(marcas[i], 0) for i in restantes]
        j = restantes.pop(int(np.argmax(aj)))
        escolhidos.append(j)
        cont[marcas[j]] = cont.get(marcas[j], 0) + 1
    return np.array(escolhidos, dtype=int)
