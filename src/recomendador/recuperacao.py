"""Etapa 1 — geração de candidatos.

- Busca densa FILTRADA (filtered ANN): a compatibilidade é aplicada dentro da busca
  vetorial (FAISS HNSW + seletor de IDs), como fazem os bancos vetoriais com
  "metadata filtering". O embedding cuida de perfil/faixa; a regra cuida da compatibilidade.
- Busca lexical BM25 (termos exatos: socket, DDR, potência…).
- Fusão híbrida por Reciprocal Rank Fusion (RRF).
"""
from __future__ import annotations

import faiss
import numpy as np

RRF_K = 60
PROFUNDIDADE = 200  # quantos itens de cada lista entram na fusão


def busca_densa_filtrada(indice: faiss.Index, vetores: np.ndarray, q: np.ndarray,
                         valido: np.ndarray, k: int) -> np.ndarray:
    """IDs dos k vizinhos mais próximos ENTRE os itens válidos (compatíveis)."""
    ids_validos = np.flatnonzero(valido)
    k = min(k, len(ids_validos))
    if k == 0:
        return np.array([], dtype=int)
    if len(ids_validos) == len(valido):
        _, I = indice.search(q, k)
    else:
        sel = faiss.IDSelectorBatch(ids_validos.astype("int64"))
        params = faiss.SearchParametersHNSW(sel=sel, efSearch=max(64, 2 * k))
        _, I = indice.search(q, k, params=params)
    ids = I[0][I[0] >= 0]
    if len(ids) < k:  # filtro muito restritivo: completa por força bruta no subconjunto
        s = vetores[ids_validos] @ q[0]
        ids = ids_validos[np.argsort(-s)[:k]]
    return ids


def ranks(scores: np.ndarray, valido: np.ndarray, profundidade: int = PROFUNDIDADE) -> np.ndarray:
    """Posição (1 = melhor) de cada item entre os válidos; inf para fora do top."""
    r = np.full(len(scores), np.inf)
    s = np.where(valido, scores, -np.inf)
    ordem = np.argsort(-s)[:profundidade]
    ordem = ordem[np.isfinite(s[ordem])]
    r[ordem] = np.arange(1, len(ordem) + 1)
    return r


def rrf(*lista_ranks: np.ndarray, k: int = RRF_K) -> np.ndarray:
    return sum(np.where(np.isfinite(r), 1.0 / (k + r), 0.0) for r in lista_ranks)


def hibrido(s_denso: np.ndarray, s_bm25: np.ndarray, valido: np.ndarray, n: int):
    """Candidatos híbridos: top-n por RRF entre a lista densa e a lista BM25."""
    r_d, r_b = ranks(s_denso, valido), ranks(s_bm25, valido)
    f = rrf(r_d, r_b)
    f = np.where(valido, f, -np.inf)
    cand = np.argsort(-f)[:n]
    cand = cand[np.isfinite(f[cand]) & (f[cand] > 0)]
    return cand, r_d, r_b, f
