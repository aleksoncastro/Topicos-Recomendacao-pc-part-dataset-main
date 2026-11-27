"""Transforma cada peça em texto para os embeddings e para os baselines lexicais.

Duas variantes:
- "bruto": só o nome e as colunas originais do dataset;
- "enriquecido": frase descritiva com os atributos derivados (socket, DDR,
  formato, faixa de preço). É a variante usada pelo sistema.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NOME_EN = {
    "cpu": "CPU processor", "motherboard": "motherboard", "memory": "RAM memory kit",
    "video-card": "graphics card GPU", "power-supply": "power supply PSU", "case": "PC case",
    "cpu-cooler": "CPU cooler", "internal-hard-drive": "storage drive SSD HDD",
}

_EXCLUIR = {"id", "marca", "categoria", "tem_preco", "outlier_preco", "pop_rank",
            "preco_pct", "faixa", "socket", "ddr", "tam_placa", "tam_max", "watts",
            "ppg_confere", "price", "name", "price_per_gb"}
_ORIGINAIS_MB = {"socket"}  # socket da placa-mãe é coluna original


def _ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and np.isnan(v)) and v != ""


def _num(v) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def texto_bruto(cat: str, r: pd.Series) -> str:
    partes = [str(r["name"])]
    for col, v in r.items():
        if col in _EXCLUIR and not (cat == "motherboard" and col in _ORIGINAIS_MB):
            continue
        if _ok(v):
            partes.append(f"{col.replace('_', ' ')} {_num(v)}")
    return ". ".join(partes)


def texto_enriquecido(cat: str, r: pd.Series) -> str:
    g = r.get
    t = f"{NOME_EN[cat]} {r['name']} by {r['marca']}, {g('faixa', '')}."
    if cat == "cpu":
        t += f" {_num(g('core_count'))} cores, {_num(g('core_clock'))} GHz base"
        if _ok(g("boost_clock")):
            t += f", {_num(g('boost_clock'))} GHz boost"
        t += f", {g('microarchitecture')} architecture"
        if _ok(g("socket")):
            t += f", socket {g('socket')}"
        if _ok(g("ddr")):
            t += f", DDR{int(g('ddr'))} memory"
        if _ok(g("tdp")):
            t += f", {_num(g('tdp'))} W TDP"
        t += ", integrated graphics" if g("gpu_integrada") else ", no integrated graphics"
    elif cat == "motherboard":
        t += f" Socket {g('socket')}, {g('form_factor')} form factor"
        if _ok(g("ddr")):
            t += f", DDR{int(g('ddr'))} memory"
        if _ok(g("memory_slots")):
            t += f", {_num(g('memory_slots'))} memory slots"
        if _ok(g("max_memory")):
            t += f", up to {_num(g('max_memory'))} GB"
    elif cat == "memory":
        if _ok(g("ddr_version")):
            t += f" DDR{int(g('ddr_version'))}"
        if _ok(g("speed_mhz")):
            t += f"-{_num(g('speed_mhz'))}"
        if _ok(g("total_gb")):
            t += f", {_num(g('total_gb'))} GB ({_num(g('n_modules'))} x {_num(g('gb_per_module'))} GB)"
        if _ok(g("cas_latency")):
            t += f", CL{_num(g('cas_latency'))}"
    elif cat == "video-card":
        t += f" {g('chipset')}"
        if _ok(g("memory")):
            t += f", {_num(g('memory'))} GB VRAM"
        if _ok(g("length")):
            t += f", {_num(g('length'))} mm long"
        t += f", about {_num(g('watts'))} W board power"
    elif cat == "power-supply":
        t += f" {_num(g('wattage'))} W {g('type')}"
        if _ok(g("efficiency")):
            t += f", 80+ {g('efficiency')}"
        if _ok(g("modular")):
            t += f", modular {g('modular')}"
    elif cat == "case":
        t += f" {g('type')}"
        if _ok(g("side_panel")):
            t += f", {g('side_panel')} side panel"
    elif cat == "cpu-cooler":
        t += f" liquid AIO {_num(g('size'))} mm radiator" if _ok(g("size")) else " air tower cooler"
        if _ok(g("noise_level_max")):
            t += f", up to {_num(g('noise_level_max'))} dB"
    elif cat == "internal-hard-drive":
        t += f" {g('drive_type')}"
        if _ok(g("capacity")):
            t += f", {_num(g('capacity'))} GB"
        if _ok(g("interface")):
            t += f", {g('interface')}"
        if _ok(g("form_factor_norm")):
            t += f", {g('form_factor_norm')}"
    if _ok(g("color")):
        t += f" Color {g('color')}."
    return t


def textos(cat: str, df: pd.DataFrame, variante: str = "enriquecido") -> list[str]:
    f = texto_enriquecido if variante == "enriquecido" else texto_bruto
    return [f(cat, r) for _, r in df.iterrows()]


def texto_consulta(alvo: str, montagem: dict[str, pd.Series], variante: str = "enriquecido") -> str:
    """A consulta é a própria montagem atual descrita em texto + a categoria pedida."""
    f = texto_enriquecido if variante == "enriquecido" else texto_bruto
    partes = [f(c, r) for c, r in montagem.items()]
    return f"{NOME_EN[alvo]} compatible with this PC build: " + " | ".join(partes)


def texto_generico(r: pd.Series) -> str:
    """Texto simples para o catálogo completo (25 categorias) do benchmark de ANN."""
    partes = [str(r.get("categoria", "")), str(r["name"])]
    for col, v in r.items():
        if col in ("id", "name", "categoria", "tem_preco", "outlier_preco"):
            continue
        if _ok(v) and not isinstance(v, (list, np.ndarray)):
            partes.append(f"{col} {_num(v)}")
    return ". ".join(partes)
