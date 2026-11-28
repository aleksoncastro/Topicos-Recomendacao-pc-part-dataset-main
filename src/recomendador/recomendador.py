"""Recomendador de peças compatíveis em duas etapas.

Etapa 1 — candidatos (velocidade e cobertura):
    montagem atual -> texto -> embedding (modelo ajustado v2)
    busca densa FILTRADA por compatibilidade (FAISS HNSW + seletor de IDs)
    + busca lexical BM25 no mesmo subconjunto
    -> fusão híbrida RRF -> top-100 candidatos
Etapa 2 — ranqueamento (ordem final):
    variáveis relacionais + neurais + lexicais + preço + popularidade
    -> LightGBM LambdaMART -> diversidade de marcas (MMR) -> top-N
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import embeddings as E
from . import ranking as R
from .baselines import Lexical
from .compat import mascara_compativel
from .dados import CATEGORIAS_MONTAGEM, carregar_catalogo
from .recuperacao import busca_densa_filtrada, hibrido
from .textos import texto_consulta, textos

ORDEM_MONTAGEM = ["motherboard", "memory", "video-card", "cpu-cooler",
                  "power-supply", "case", "internal-hard-drive"]


@dataclass
class Recomendador:
    versao_modelo: str = "v2"      # "base", "v1" ou "v2"
    k_candidatos: int = 100
    mmr: float = 0.3
    usar_ranker: bool = True

    def __post_init__(self):
        if self.versao_modelo != "base" and not E.caminho_modelo(self.versao_modelo).exists():
            print(f"Aviso: modelo {self.versao_modelo} não encontrado; usando o modelo base.")
            self.versao_modelo = "base"
        self.modelo_nome = E.nome_modelo(self.versao_modelo)
        self.cat = carregar_catalogo()
        self.textos = {c: textos(c, self.cat[c], "enriquecido") for c in CATEGORIAS_MONTAGEM}
        self.vetores, self.indices = {}, {}
        for c in CATEGORIAS_MONTAGEM:
            v = E.embed(self.textos[c], f"{c}_enriquecido_{self.versao_modelo}",
                        modelo_nome=self.modelo_nome)
            self.vetores[c] = v
            self.indices[c] = E.indice_hnsw(v)
        self.lex = Lexical(self.textos)
        self.ranker = R.carregar() if self.usar_ranker else None

    # ---------------- etapa 1 ----------------
    def vetor_consulta(self, alvo: str, montagem: dict) -> np.ndarray:
        return E.embed([texto_consulta(alvo, montagem)], modelo_nome=self.modelo_nome)

    def candidatos(self, alvo: str, montagem: dict, filtrar: bool = True) -> dict:
        df = self.cat[alvo]
        valido = mascara_compativel(alvo, df, montagem) if filtrar else np.ones(len(df), bool)
        q = self.vetor_consulta(alvo, montagem)
        # densa filtrada (ANN) -> scores exatos dos itens recuperados
        ids_ann = busca_densa_filtrada(self.indices[alvo], self.vetores[alvo], q, valido,
                                       2 * self.k_candidatos)
        s_denso = np.full(len(df), -np.inf)
        s_denso[ids_ann] = self.vetores[alvo][ids_ann] @ q[0]
        s_bm25 = self.lex.bm25_scores(alvo, texto_consulta(alvo, montagem))
        cand, r_d, r_b, f = hibrido(s_denso, s_bm25, valido, self.k_candidatos)
        s_denso_full = self.vetores[alvo] @ q[0]  # para as variáveis do ranker
        return {"cand": cand, "s_denso": s_denso_full, "s_bm25": s_bm25,
                "r_denso": r_d, "r_bm25": r_b, "rrf": f, "valido": valido}

    # ---------------- etapa 2 ----------------
    def pontuar(self, alvo: str, montagem: dict, c: dict) -> tuple[np.ndarray, pd.DataFrame]:
        X = R.montar_features(alvo, self.cat[alvo], c["cand"], montagem, c["s_denso"],
                              c["s_bm25"], c["r_denso"], c["r_bm25"], c["rrf"])
        if self.ranker is not None:
            return self.ranker.predict(X[self.ranker.features_]), X
        return c["rrf"][c["cand"]], X  # sem ranker treinado: ordem do RRF

    def recomendar(self, alvo: str, montagem: dict, n: int = 10) -> pd.DataFrame:
        c = self.candidatos(alvo, montagem)
        if len(c["cand"]) == 0:
            return self.cat[alvo].iloc[[]]
        s, _ = self.pontuar(alvo, montagem, c)
        ordem = R.mmr_marcas(s, self.cat[alvo]["marca"].to_numpy()[c["cand"]], n, self.mmr)
        out = self.cat[alvo].iloc[c["cand"][ordem]].copy()
        out["similaridade"] = c["s_denso"][c["cand"][ordem]]
        out["score"] = s[ordem]
        return out

    # ---------------- utilidades ----------------
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
