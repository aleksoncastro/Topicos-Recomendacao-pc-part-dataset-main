"""Geração (com cache em disco) dos embeddings e construção dos índices FAISS."""
from __future__ import annotations

import hashlib
from pathlib import Path

import faiss
import numpy as np

from .dados import RAIZ

MODELO_BASE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODELO_AJUSTADO = RAIZ / "modelos" / "minilm-pc-compat"
CACHE = RAIZ / "cache"
_modelos: dict = {}


def nome_modelo(ajustado: bool) -> str:
    return str(MODELO_AJUSTADO) if ajustado else MODELO_BASE


def modelo(nome: str = MODELO_BASE):
    if nome not in _modelos:
        from sentence_transformers import SentenceTransformer
        _modelos[nome] = SentenceTransformer(nome, device="cpu")
    return _modelos[nome]


def embed(textos: list[str], nome_cache: str | None = None, batch: int = 128,
          modelo_nome: str = MODELO_BASE) -> np.ndarray:
    """Vetores normalizados (produto interno = similaridade de cosseno)."""
    if nome_cache:
        h = hashlib.md5(("\n".join(textos) + modelo_nome).encode()).hexdigest()[:10]
        arq = CACHE / f"{nome_cache}_{h}.npy"
        if arq.exists():
            return np.load(arq)
    v = modelo(modelo_nome).encode(textos, batch_size=batch, normalize_embeddings=True,
                                   show_progress_bar=len(textos) > 2000, convert_to_numpy=True)
    v = v.astype("float32")
    if nome_cache:
        CACHE.mkdir(exist_ok=True)
        np.save(arq, v)
    return v


# ---------------------------------------------------------------------------
# Índices
# ---------------------------------------------------------------------------
def indice_exato(x: np.ndarray) -> faiss.Index:
    idx = faiss.IndexFlatIP(x.shape[1])
    idx.add(x)
    return idx


def indice_hnsw(x: np.ndarray, M: int = 32, ef_construcao: int = 200, ef_busca: int = 64) -> faiss.Index:
    idx = faiss.IndexHNSWFlat(x.shape[1], M, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = ef_construcao
    idx.add(x)
    idx.hnsw.efSearch = ef_busca
    return idx


def indice_ivf(x: np.ndarray, nlist: int | None = None, nprobe: int = 8) -> faiss.Index:
    nlist = nlist or max(4, int(4 * np.sqrt(len(x))))
    q = faiss.IndexFlatIP(x.shape[1])
    idx = faiss.IndexIVFFlat(q, x.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
    idx.train(x)
    idx.add(x)
    idx.nprobe = nprobe
    return idx


def indice_ivfpq(x: np.ndarray, nlist: int | None = None, m: int = 48, nprobe: int = 8) -> faiss.Index:
    nlist = nlist or max(4, int(4 * np.sqrt(len(x))))
    q = faiss.IndexFlatIP(x.shape[1])
    idx = faiss.IndexIVFPQ(q, x.shape[1], nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
    idx.train(x)
    idx.add(x)
    idx.nprobe = nprobe
    return idx


def tamanho_mb(idx: faiss.Index) -> float:
    return len(faiss.serialize_index(idx)) / 1e6
