"""Fine-tuning do modelo de embeddings (aprendizado contrastivo).

Ensina o modelo que "montagem X" fica perto de peças compatíveis e de faixa coerente,
e longe das incompatíveis. Usa SÓ as CPUs do grupo "ajuste" (plataformas antigas);
AM5, LGA1700 e LGA1851 ficam fora para medir generalização.

Duas receitas (para comparar no relatório):
  v1 — 3 plataformas (LGA1151, AM4, LGA1200), 1 negativo difícil "semântico".
       Resultado: decora os valores de socket e não generaliza.
  v2 — todas as plataformas antigas (16 sockets) +
       anonimização: com prob. 0,5 cada socket/DDR do exemplo vira um código aleatório
       (o mesmo na âncora e nas peças), forçando o modelo a aprender a REGRA de igualdade
       em vez de decorar valores +
       par mínimo: um 2º negativo incompatível com preço quase igual ao do positivo
       (ex.: placa B650 x B550 na mesma faixa), que só difere no que decide a compatibilidade.

Uso:  python -m src.recomendador.finetune [--versao v2]
"""
from __future__ import annotations

import argparse
import random
import re
import string
import time

import numpy as np
from datasets import Dataset

from . import embeddings as E
from .consultas import ALVOS, PLAT_TESTE, gerar_consultas
from .dados import RAIZ, carregar_catalogo
from .textos import texto_consulta, textos

POS_POR_CONSULTA = 4
PLAT_V1 = {"LGA1151", "AM4", "LGA1200"}


# ---------------------------------------------------------------------------
# Anonimização de socket / DDR
# ---------------------------------------------------------------------------
def _padrao_valores(cat) -> re.Pattern:
    sockets = set(cat["cpu"]["socket"].dropna()) | set(cat["motherboard"]["socket"].dropna())
    partes = {p for s in sockets for p in str(s).replace("2 x ", "").split("/")}
    partes = sorted(partes, key=len, reverse=True)  # mais longos primeiro (LGA2011-3 antes de LGA2011)
    alt = "|".join(re.escape(p) for p in partes)
    return re.compile(rf"(?<![A-Za-z0-9])({alt}|DDR[2-5])(?![A-Za-z0-9+])")


def anonimizar(textos_: list[str], padrao: re.Pattern, rnd: random.Random) -> list[str]:
    """Troca cada valor distinto por um código aleatório, consistente entre os textos."""
    mapa: dict[str, str] = {}

    def troca(m):
        v = m.group(1)
        if v not in mapa:
            pre = "MEM" if v.startswith("DDR") else "SKT"
            mapa[v] = pre + "".join(rnd.choices(string.ascii_uppercase + string.digits, k=3))
        return mapa[v]

    return [padrao.sub(troca, t) for t in textos_]


# ---------------------------------------------------------------------------
# Exemplos de treino
# ---------------------------------------------------------------------------
def montar_exemplos(versao: str, seed: int = 7):
    rnd = random.Random(seed)
    cat = carregar_catalogo()
    padrao = _padrao_valores(cat)
    txt = {c: textos(c, cat[c], "enriquecido") for c in ALVOS}
    vets = {c: E.embed(txt[c], f"{c}_enriquecido_base") for c in ALVOS}
    qs = [q for q in gerar_consultas(cat) if q["split"] == "ajuste"]
    if versao == "v1":
        qs = [q for q in qs if q["socket"] in PLAT_V1]
    linhas = {"anchor": [], "positive": [], "negative": []}
    if versao == "v2":
        linhas["negative_2"] = []
    n_anon = 0
    for q in qs:
        alvo, g = q["alvo"], q["grades"]
        df = cat[alvo]
        tq = texto_consulta(alvo, q["montagem"])
        bons = np.flatnonzero(g == 2)
        if len(bons) == 0:
            bons = np.flatnonzero(g == 1)
        # negativo semântico: incompatível mas parecido segundo o modelo base
        ordem = np.argsort(-(vets[alvo] @ E.embed([tq])[0]))
        # itens das plataformas de teste nunca entram no treino (nem como negativos)
        permitido = (~df["socket"].isin(PLAT_TESTE)).to_numpy() if alvo == "motherboard"             else np.ones(len(df), dtype=bool)
        dificeis = ([i for i in ordem[:60] if g[i] == 0 and permitido[i]]
                    or list(np.flatnonzero((g == 0) & permitido))
                    or [i for i in ordem[:60] if g[i] == 1])
        incompat = np.flatnonzero((g == 0) & permitido)
        if not dificeis:
            continue
        for p in rnd.sample(list(bons), min(POS_POR_CONSULTA, len(bons))):
            n1 = rnd.choice(dificeis)
            trio = [tq, txt[alvo][p], txt[alvo][n1]]
            if versao == "v2":
                # par mínimo: incompatível com preço mais próximo do positivo
                if len(incompat):
                    dist = np.abs(df["preco_pct"].to_numpy()[incompat] - df["preco_pct"].iloc[p])
                    n2 = int(incompat[rnd.choice(list(np.argsort(dist)[:5]))])
                else:
                    n2 = rnd.choice(dificeis)
                trio.append(txt[alvo][n2])
                if rnd.random() < 0.5:
                    trio = anonimizar(trio, padrao, rnd)
                    n_anon += 1
            for col, t in zip(linhas.keys(), trio):
                linhas[col].append(t)
    return Dataset.from_dict(linhas), len(qs), n_anon


def main():
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    ap = argparse.ArgumentParser()
    ap.add_argument("--versao", choices=["v1", "v2"], default="v2")
    ap.add_argument("--epocas", type=int, default=2)
    a = ap.parse_args()

    t0 = time.time()
    ds, nq, n_anon = montar_exemplos(a.versao)
    print(f"[{a.versao}] {len(ds)} exemplos de {nq} consultas de ajuste "
          f"({n_anon} anonimizados) | colunas: {ds.column_names}")
    print("Exemplo:", {k: v[:160] for k, v in ds[0].items()})
    model = SentenceTransformer(E.MODELO_BASE, device="cpu")
    model.max_seq_length = 192
    args = SentenceTransformerTrainingArguments(
        output_dir=str(RAIZ / "cache" / "_treino_tmp"),
        num_train_epochs=a.epocas,
        per_device_train_batch_size=32,
        learning_rate=3e-5,
        warmup_steps=0.1,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no",
        logging_steps=20,
        report_to="none",
        seed=42,
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds,
                                         loss=MultipleNegativesRankingLoss(model))
    trainer.train()
    destino = E.caminho_modelo(a.versao)
    model.save(str(destino))
    print(f"Modelo salvo em {destino} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
