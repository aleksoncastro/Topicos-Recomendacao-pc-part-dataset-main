"""Fine-tuning do modelo de embeddings (aprendizado contrastivo).

Ideia: ensinar o modelo que "montagem X" fica perto de peças compatíveis e
de faixa coerente, e longe das incompatíveis. Usa SÓ as CPUs do grupo "ajuste"
(plataformas antigas: LGA1151, AM4, LGA1200). As plataformas novas
(AM5, LGA1700, LGA1851) ficam fora do treino para medir generalização.

Uso:  python -m src.finetune
"""
from __future__ import annotations

import random
import time

import numpy as np
from datasets import Dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from . import embeddings as E
from .avaliacao import ALVOS, gerar_consultas
from .dados import RAIZ, carregar_catalogo
from .textos import texto_consulta, textos

POS_POR_CONSULTA = 4


def montar_exemplos(seed: int = 7):
    rnd = random.Random(seed)
    cat = carregar_catalogo()
    txt = {c: textos(c, cat[c], "enriquecido") for c in ALVOS}
    vets = {c: E.embed(txt[c], f"{c}_enriquecido_base") for c in ALVOS}
    qs = [q for q in gerar_consultas(cat) if q["split"] == "ajuste"]
    anc, pos, neg = [], [], []
    for q in qs:
        alvo, g = q["alvo"], q["grades"]
        tq = texto_consulta(alvo, q["montagem"])
        bons = np.flatnonzero(g == 2)
        if len(bons) == 0:
            bons = np.flatnonzero(g == 1)
        # negativo "difícil": incompatível mas parecido segundo o modelo base
        sims = vets[alvo] @ E.embed([tq])[0]
        ordem = np.argsort(-sims)
        dificeis = ([i for i in ordem[:60] if g[i] == 0] or list(np.flatnonzero(g == 0))
                    or [i for i in ordem[:60] if g[i] == 1])
        if not dificeis:
            continue
        for p in rnd.sample(list(bons), min(POS_POR_CONSULTA, len(bons))):
            anc.append(tq)
            pos.append(txt[alvo][p])
            neg.append(txt[alvo][rnd.choice(dificeis)])
    return Dataset.from_dict({"anchor": anc, "positive": pos, "negative": neg}), len(qs)


def main():
    t0 = time.time()
    ds, nq = montar_exemplos()
    print(f"{len(ds)} triplas (âncora, positivo, negativo) de {nq} consultas de ajuste")
    model = SentenceTransformer(E.MODELO_BASE, device="cpu")
    model.max_seq_length = 256
    args = SentenceTransformerTrainingArguments(
        output_dir=str(RAIZ / "modelos" / "_tmp"),
        num_train_epochs=2,
        per_device_train_batch_size=32,
        learning_rate=3e-5,
        warmup_ratio=0.1,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no",
        logging_steps=20,
        report_to="none",
        seed=42,
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds,
                                         loss=MultipleNegativesRankingLoss(model))
    trainer.train()
    model.save(str(E.MODELO_AJUSTADO))
    print(f"Modelo salvo em {E.MODELO_AJUSTADO} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
