"""Recomendador de peças compatíveis em duas etapas.

Etapa 1 — geração de candidatos (neural): a montagem atual vira texto -> embedding,
          e a busca ANN (HNSW) traz os K itens mais próximos da categoria pedida.
Etapa 2 — ranqueamento: filtro de compatibilidade (restrição do negócio) +
          combinação de similaridade, popularidade e coerência de preço +
          diversidade de marcas (MMR).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import embeddings as E
from .compat import mascara_compativel
from .dados import CATEGORIAS_MONTAGEM, RAIZ, carregar_catalogo
from .textos import texto_consulta, textos

ORDEM_MONTAGEM = ["motherboard", "memory", "video-card", "cpu-cooler",
                  "power-supply", "case", "internal-hard-drive"]


@dataclass
class Pesos:
    sim: float = 1.0
    pop: float = 0.5
    coer: float = 1.0
    outlier: float = 0.5
    mmr: float = 0.3


def ancora(montagem: dict[str, pd.Series]) -> pd.Series:
    return montagem.get("cpu", next(iter(montagem.values())))


def features(df: pd.DataFrame, cand: np.ndarray, sims: np.ndarray, anc: pd.Series) -> dict:
    sub = df.iloc[cand]
    z = (sims - sims.mean()) / (sims.std() + 1e-9)
    return {
        "sim": z,
        "pop": 1.0 - sub["pop_rank"].to_numpy(),
        "coer": 1.0 - np.abs(sub["preco_pct"].to_numpy() - anc["preco_pct"]),
        "outlier": sub["outlier_preco"].to_numpy().astype(float),
        "marca": sub["marca"].to_numpy(),
    }


def ordenar(f: dict, p: Pesos, n: int, valido: np.ndarray | None = None) -> np.ndarray:
    """Devolve as posições (dentro dos candidatos) na ordem final, com MMR por marca."""
    s = p.sim * f["sim"] + p.pop * f["pop"] + p.coer * f["coer"] - p.outlier * f["outlier"]
    if valido is not None:
        s = np.where(valido, s, -np.inf)
    escolhidos: list[int] = []
    cont: dict[str, int] = {}
    restantes = np.argsort(-s)[: max(n * 5, 50)]
    restantes = [i for i in restantes if np.isfinite(s[i])]
    while restantes and len(escolhidos) < n:
        ajust = [s[i] - p.mmr * cont.get(f["marca"][i], 0) for i in restantes]
        j = restantes.pop(int(np.argmax(ajust)))
        escolhidos.append(j)
        cont[f["marca"][j]] = cont.get(f["marca"][j], 0) + 1
    return np.array(escolhidos, dtype=int)


@dataclass
class Recomendador:
    variante: str = "enriquecido"
    tipo_indice: str = "hnsw"
    k_candidatos: int = 100
    ajustado: bool = True          # usa o modelo com fine-tuning, se existir
    pesos: Pesos | None = None     # None = carrega resultados/pesos.json (ajustados no treino)

    def __post_init__(self):
        if self.ajustado and not E.MODELO_AJUSTADO.exists():
            print("Aviso: modelo ajustado não encontrado; usando o modelo base.")
            self.ajustado = False
        self.modelo_nome = E.nome_modelo(self.ajustado)
        if self.pesos is None:
            arq = RAIZ / "resultados" / "pesos.json"
            self.pesos = Pesos(**json.loads(arq.read_text())) if arq.exists() else Pesos()
        self.cat = carregar_catalogo()
        self.vetores, self.indices, self.exatos = {}, {}, {}
        sufixo = "ft" if self.ajustado else "base"
        for c in CATEGORIAS_MONTAGEM:
            v = E.embed(textos(c, self.cat[c], self.variante), f"{c}_{self.variante}_{sufixo}",
                        modelo_nome=self.modelo_nome)
            self.vetores[c] = v
            self.exatos[c] = E.indice_exato(v)
            self.indices[c] = E.indice_hnsw(v) if self.tipo_indice == "hnsw" else self.exatos[c]

    # ---------------- etapa 1 ----------------
    def vetor_consulta(self, alvo: str, montagem: dict[str, pd.Series]) -> np.ndarray:
        return E.embed([texto_consulta(alvo, montagem, self.variante)], modelo_nome=self.modelo_nome)

    def candidatos(self, alvo: str, montagem: dict[str, pd.Series], k: int | None = None,
                   exato: bool = False):
        k = min(k or self.k_candidatos, len(self.cat[alvo]))
        idx = self.exatos[alvo] if exato else self.indices[alvo]
        sims, ids = idx.search(self.vetor_consulta(alvo, montagem), k)
        ok = ids[0] >= 0
        return ids[0][ok], sims[0][ok]

    # ---------------- etapa 2 ----------------
    def recomendar(self, alvo: str, montagem: dict[str, pd.Series], n: int = 10,
                   filtrar: bool = True) -> pd.DataFrame:
        df = self.cat[alvo]
        cand, sims = self.candidatos(alvo, montagem)
        valido = mascara_compativel(alvo, df, montagem)[cand] if filtrar else None
        f = features(df, cand, sims, ancora(montagem))
        ordem = ordenar(f, self.pesos, n, valido)
        out = df.iloc[cand[ordem]].copy()
        out["similaridade"] = sims[ordem]
        return out

    def buscar_texto(self, consulta: str, categoria: str = "cpu", n: int = 5) -> pd.DataFrame:
        """Busca livre (aceita português): ex. 'processador AMD barato para jogos'."""
        sims, ids = self.indices[categoria].search(E.embed([consulta], modelo_nome=self.modelo_nome), n)
        out = self.cat[categoria].iloc[ids[0]].copy()
        out["similaridade"] = sims[0]
        return out

    def encontrar(self, categoria: str, nome: str) -> pd.Series:
        df = self.cat[categoria]
        padrao = r"(?:^|\s)" + re.escape(nome.lower()) + r"$"
        exato = df[df["name"].str.lower().str.contains(padrao, regex=True)]
        if len(exato):
            return exato.loc[exato["name"].str.len().idxmin()]
        parcial = df[df["name"].str.lower().str.contains(nome.lower(), regex=False)]
        if len(parcial):
            return parcial.loc[parcial["name"].str.len().idxmin()]
        return self.buscar_texto(nome, categoria, 1).iloc[0]

    def montar(self, inicial: pd.Series, categoria_inicial: str = "cpu", n: int = 5,
               ordem: list[str] | None = None):
        """Monta o PC passo a passo: a cada etapa escolhe o 1º recomendado e segue."""
        montagem = {categoria_inicial: inicial}
        passos = []
        for alvo in ordem or ORDEM_MONTAGEM:
            if alvo in montagem:
                continue
            recs = self.recomendar(alvo, montagem, n)
            if recs.empty:
                continue
            passos.append((alvo, recs))
            montagem[alvo] = recs.iloc[0]
        return montagem, passos
