"""Regras de compatibilidade e definição de relevância graduada.

Relevância de um item candidato para uma montagem:
  0 = incompatível com a montagem
  1 = compatível
  2 = compatível E na mesma faixa de preço da CPU dentro da sua plataforma
      (|percentil do item entre os compatíveis - percentil da CPU no seu socket| <= 0,25)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dados import potencia_recomendada

TOL_FAIXA = 0.25


def _aceita_socket(sockets_placa: pd.Series, socket_cpu) -> np.ndarray:
    """Placas como "AM3/AM2+/AM2" aceitam mais de um socket."""
    return sockets_placa.fillna("").str.split("/").map(lambda l: socket_cpu in l).to_numpy()


def mascara_compativel(alvo: str, df: pd.DataFrame, montagem: dict[str, pd.Series]) -> np.ndarray:
    ok = np.ones(len(df), dtype=bool)
    cpu = montagem.get("cpu")
    mb = montagem.get("motherboard")
    mem = montagem.get("memory")
    gab = montagem.get("case")
    gpu = montagem.get("video-card")
    psu = montagem.get("power-supply")

    if alvo == "cpu" and mb is not None:
        ok &= df["socket"].isin(str(mb["socket"]).split("/")).to_numpy()

    if alvo == "motherboard":
        if cpu is not None:
            ok &= _aceita_socket(df["socket"], cpu["socket"])
            if not pd.isna(cpu.get("ddr")):
                ok &= (df["ddr"] == cpu["ddr"]).to_numpy()
        if gab is not None:
            ok &= (df["tam_placa"] <= gab["tam_max"]).to_numpy()
        if mem is not None:
            ok &= (df["ddr"] == mem["ddr_version"]).to_numpy()
            ok &= (df["max_memory"].fillna(1e9) >= mem["total_gb"]).to_numpy()
            ok &= (df["memory_slots"].fillna(99) >= mem["n_modules"]).to_numpy()

    if alvo == "memory":
        if mb is not None:
            ok &= (df["ddr_version"] == mb["ddr"]).to_numpy()
            if not pd.isna(mb.get("max_memory")):
                ok &= (df["total_gb"] <= mb["max_memory"]).to_numpy()
            if not pd.isna(mb.get("memory_slots")):
                ok &= (df["n_modules"] <= mb["memory_slots"]).to_numpy()
        elif cpu is not None and not pd.isna(cpu.get("ddr")):
            ok &= (df["ddr_version"] == cpu["ddr"]).to_numpy()

    if alvo == "case":
        if mb is not None:
            ok &= (df["tam_max"] >= mb["tam_placa"]).to_numpy()
        if psu is not None and psu.get("type") == "ATX":
            ok &= (df["tam_max"] >= 2).to_numpy()  # Mini ITX costuma exigir SFX

    if alvo == "cpu-cooler" and cpu is not None:
        # coolers "box" da Intel/AMD só servem na CPU da mesma marca
        marca = df["marca"].to_numpy()
        ok &= ~np.isin(marca, ["Intel", "AMD"]) | (marca == cpu["marca"])

    if alvo == "power-supply":
        tdp = cpu["tdp"] if cpu is not None else np.nan
        w_gpu = gpu["watts"] if gpu is not None else 0.0
        ok &= (df["wattage"] >= potencia_recomendada(tdp, w_gpu)).to_numpy()
        if gab is not None and gab.get("tam_max") == 1:
            ok &= df["type"].isin(["SFX", "Mini ITX", "Flex ATX"]).to_numpy()

    return ok


def percentil_cpu_na_plataforma(cpus: pd.DataFrame) -> pd.Series:
    return cpus.groupby("socket")["price"].rank(pct=True)


def relevancia(alvo: str, df: pd.DataFrame, montagem: dict[str, pd.Series],
               pct_ancora: float) -> np.ndarray:
    comp = mascara_compativel(alvo, df, montagem)
    g = comp.astype(int)
    if comp.sum() > 0:
        pct = df.loc[comp, "price"].rank(pct=True).to_numpy()
        coerente = np.abs(pct - pct_ancora) <= TOL_FAIXA
        idx = np.flatnonzero(comp)
        g[idx[coerente]] = 2
    return g
