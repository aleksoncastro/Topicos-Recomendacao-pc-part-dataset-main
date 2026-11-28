"""Consultas de avaliação e divisão por geração de plataforma.

O dataset não tem datas, então a "divisão temporal" usa a geração do socket:
- treino: todas as plataformas lançadas até 2020 (ajuste 70% / validação 30% das CPUs)
- teste:  plataformas lançadas de 2021 em diante (AM5, LGA1700, LGA1851),
          que nenhum modelo vê durante o treino.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from .compat import mascara_compativel, percentil_cpu_na_plataforma, relevancia

ALVOS = ["motherboard", "memory", "case", "power-supply"]
PLAT_TESTE = ["LGA1700", "AM5", "LGA1851"]
PLAT_TREINO = ["LGA775", "LGA1156", "LGA1366", "LGA1155", "LGA2011", "LGA1150",
               "LGA2011-3", "LGA1151", "LGA2066", "LGA1200",
               "AM3", "AM3+", "FM1", "FM2", "FM2+", "AM4"]


def grupo_da_cpu(nome: str, socket: str) -> str:
    if socket in PLAT_TESTE:
        return "teste"
    return "ajuste" if zlib.crc32(nome.encode()) % 10 < 7 else "validacao"


def gerar_consultas(cat: dict[str, pd.DataFrame], seed: int = 42) -> list[dict]:
    """Para cada CPU: placa-mãe | CPU; memória e gabinete | CPU + placa; fonte | CPU + GPU."""
    rng = np.random.default_rng(seed)
    cpus = cat["cpu"]
    cpus = cpus[cpus["socket"].isin(PLAT_TREINO + PLAT_TESTE)].drop_duplicates("name").copy()
    cpus["pct_plat"] = percentil_cpu_na_plataforma(cpus)
    gpus = cat["video-card"]
    qs = []
    for _, cpu in cpus.iterrows():
        grupo = grupo_da_cpu(cpu["name"], cpu["socket"])
        pct = float(cpu["pct_plat"])
        mbs = cat["motherboard"][mascara_compativel("motherboard", cat["motherboard"], {"cpu": cpu})]
        if mbs.empty:
            continue
        mb = mbs.iloc[rng.integers(len(mbs))]
        perto = (gpus["preco_pct"] - cpu["preco_pct"]).abs().nsmallest(10).index
        gpu = gpus.loc[rng.choice(perto)]
        for alvo, mont in [("motherboard", {"cpu": cpu}),
                           ("memory", {"cpu": cpu, "motherboard": mb}),
                           ("case", {"cpu": cpu, "motherboard": mb}),
                           ("power-supply", {"cpu": cpu, "video-card": gpu})]:
            g = relevancia(alvo, cat[alvo], mont, pct)
            if (g > 0).sum() == 0:
                continue
            qs.append({"alvo": alvo, "montagem": mont, "pct": pct, "split": grupo,
                       "cpu": cpu["name"], "socket": cpu["socket"], "grades": g})
    return qs
