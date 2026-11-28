"""Carregamento do catálogo limpo e derivação dos atributos de compatibilidade.

O dataset não traz o socket das CPUs nem o tipo de memória das placas-mãe, então
esses atributos são derivados por regras documentadas aqui (ver relatório, seção 2).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]  # raiz do projeto
DADOS = RAIZ / "data" / "processed"

# Categorias que formam uma montagem
CATEGORIAS_MONTAGEM = [
    "cpu", "motherboard", "memory", "video-card",
    "power-supply", "case", "cpu-cooler", "internal-hard-drive",
]

NOME_PT = {
    "cpu": "processador", "motherboard": "placa-mãe", "memory": "memória RAM",
    "video-card": "placa de vídeo", "power-supply": "fonte", "case": "gabinete",
    "cpu-cooler": "cooler", "internal-hard-drive": "armazenamento",
}

# ---------------------------------------------------------------------------
# CPU -> socket
# ---------------------------------------------------------------------------
_SOCKET_POR_ARQ = {
    "Zen": "AM4", "Zen+": "AM4", "Zen 2": "AM4", "Zen 3": "AM4",
    "Zen 4": "AM5", "Zen 5": "AM5",
    "Excavator": "FM2+", "Steamroller": "FM2+", "Bulldozer": "AM3+",
    "K10": "AM3", "Lynx": "FM1", "Jaguar": "AM1", "Puma+": "AM1",
    "Core": "LGA775", "Wolfdale": "LGA775", "Yorkfield": "LGA775",
    "Nehalem": "LGA1156", "Westmere": "LGA1156",
    "Sandy Bridge": "LGA1155", "Ivy Bridge": "LGA1155",
    "Haswell": "LGA1150", "Haswell Refresh": "LGA1150", "Broadwell": "LGA1150",
    "Skylake": "LGA1151", "Kaby Lake": "LGA1151",
    "Coffee Lake": "LGA1151", "Coffee Lake Refresh": "LGA1151",
    "Comet Lake": "LGA1200", "Rocket Lake": "LGA1200",
    "Alder Lake": "LGA1700", "Raptor Lake": "LGA1700", "Raptor Lake Refresh": "LGA1700",
    "Arrow Lake": "LGA1851", "Cascade Lake": "LGA2066",
}


def socket_cpu(nome: str, arq: str | None) -> str | None:
    """Deriva o socket a partir da microarquitetura, com exceções pelo nome."""
    if arq is None or (isinstance(arq, float) and np.isnan(arq)):
        return None
    n = nome
    if "Threadripper" in n or "EPYC" in n or "Opteron" in n:
        return None  # plataformas de servidor/HEDT fora do escopo
    if "Xeon E5" in n:
        return {"Sandy Bridge": "LGA2011", "Ivy Bridge": "LGA2011"}.get(arq, "LGA2011-3")
    if arq == "Piledriver":
        return "AM3+" if "FX-" in n else "FM2"
    if arq == "Nehalem" and re.search(r"i7-9\d\d", n):
        return "LGA1366"
    if arq == "Westmere" and re.search(r"i7-9\d\d", n):
        return "LGA1366"
    if arq in ("Sandy Bridge", "Ivy Bridge") and re.search(r"i7-(38|39|48|49)\d\d", n):
        return "LGA2011"
    if arq in ("Haswell", "Broadwell") and re.search(r"i7-(58|59|68|69)\d\d", n):
        return "LGA2011-3"
    if arq in ("Skylake", "Kaby Lake") and re.search(r"i\d-\d{4,5}X", n):
        return "LGA2066"
    return _SOCKET_POR_ARQ.get(arq)


# Geração de DDR suportada por socket
_DDR_POR_SOCKET = {
    "AM5": 5, "LGA1851": 5,
    "AM4": 4, "LGA1200": 4, "LGA1151": 4, "LGA2066": 4, "LGA2011-3": 4,
    "LGA1150": 3, "LGA1155": 3, "LGA1156": 3, "LGA1366": 3, "LGA2011": 3,
    "AM3+": 3, "AM3": 3, "FM2": 3, "FM2+": 3, "FM1": 3, "AM1": 3,
}


def ddr_placa(socket: str | None, nome: str) -> float:
    if socket == "LGA1700":
        return 4.0 if re.search(r"DDR4|/D4\b|\bD4\b", nome) else 5.0
    v = _DDR_POR_SOCKET.get(str(socket or "").split("/")[0])
    return float(v) if v else np.nan


# ---------------------------------------------------------------------------
# Formatos: placa-mãe x gabinete
# ---------------------------------------------------------------------------
def tamanho_placa(ff: str | None) -> float:
    if not isinstance(ff, str):
        return np.nan
    if "ITX" in ff or "DTX" in ff:
        return 1
    if ff in ("Micro ATX", "Flex ATX"):
        return 2
    if ff == "ATX":
        return 3
    return 4  # EATX, XL ATX, SSI, HPTX


def tamanho_max_gabinete(tipo: str | None) -> float:
    if not isinstance(tipo, str):
        return np.nan
    if tipo.startswith("Mini ITX"):
        return 1
    if tipo.startswith("MicroATX") or tipo == "HTPC":
        return 2
    if tipo == "ATX Full Tower":
        return 4
    if tipo.startswith("ATX"):
        return 3
    return np.nan


# ---------------------------------------------------------------------------
# Consumo estimado de GPU (W) — tabela aproximada de TBP por chipset
# ---------------------------------------------------------------------------
_GPU_W = [
    ("5090", 575), ("5080", 360), ("5070 Ti", 300), ("5070", 250), ("5060 Ti", 180), ("5060", 145),
    ("4090", 450), ("4080", 320), ("4070 Ti", 285), ("4070", 200), ("4060 Ti", 165), ("4060", 115),
    ("3090 Ti", 450), ("3090", 350), ("3080 Ti", 350), ("3080", 320), ("3070 Ti", 290),
    ("3070", 220), ("3060 Ti", 200), ("3060", 170), ("3050", 130),
    ("2080 Ti", 250), ("2080", 215), ("2070", 175), ("2060", 160),
    ("1660", 120), ("1650", 75), ("1080 Ti", 250), ("1080", 180), ("1070", 150), ("1060", 120),
    ("1050", 75), ("980", 165), ("970", 145), ("960", 120),
    ("9070 XT", 304), ("9070", 220), ("9060 XT", 160),
    ("7900 XTX", 355), ("7900 XT", 315), ("7900 GRE", 260), ("7800 XT", 263), ("7700 XT", 245),
    ("7600", 165), ("6950", 335), ("6900", 300), ("6800 XT", 300), ("6800", 250), ("6750", 250),
    ("6700", 230), ("6650", 180), ("6600", 132), ("6500", 107), ("6400", 53),
    ("5700", 225), ("5600", 150), ("5500", 130), ("590", 225), ("580", 185), ("570", 150),
    ("560", 75), ("550", 50), ("Vega 64", 295), ("Vega 56", 210),
    ("B580", 190), ("B570", 150), ("A770", 225), ("A750", 225), ("A380", 75),
]


def consumo_gpu(chipset: str | None) -> float:
    if not isinstance(chipset, str):
        return 200.0
    for chave, w in _GPU_W:
        if chave in chipset:
            return float(w)
    return 200.0


def potencia_recomendada(tdp_cpu: float, w_gpu: float = 0.0) -> float:
    """Regra prática: (CPU + GPU + 75 W do resto do sistema) x 1,3 de folga."""
    tdp = tdp_cpu if tdp_cpu and not np.isnan(tdp_cpu) else 95.0
    return (tdp + w_gpu + 75.0) * 1.3


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------
def _faixa(pct: float) -> str:
    if np.isnan(pct):
        return "unknown price"
    if pct < 0.25:
        return "budget"
    if pct < 0.5:
        return "mid-range"
    if pct < 0.8:
        return "upper mid-range"
    return "high-end"


def carregar_categoria(cat: str, somente_com_preco: bool = True) -> pd.DataFrame:
    df = pd.read_parquet(DADOS / f"{cat}.parquet").reset_index(drop=True)
    # A ordem original do scraper segue a ordenação padrão do PCPartPicker
    # (popularidade). Usamos isso como sinal de popularidade (0 = mais popular).
    df["pop_rank"] = np.arange(len(df)) / max(len(df) - 1, 1)
    if somente_com_preco:
        df = df[df["tem_preco"]].reset_index(drop=True)
    df["preco_pct"] = df["price"].rank(pct=True)
    df["faixa"] = df["preco_pct"].map(_faixa)

    if cat == "cpu":
        df["socket"] = [socket_cpu(n, a) for n, a in zip(df["name"], df["microarchitecture"])]
        df["ddr"] = [ddr_placa(s, "") if s != "LGA1700" else np.nan for s in df["socket"]]
        # faixa da CPU dentro da própria plataforma (0 = mais barata do socket, 1 = mais cara)
        df["pct_plataforma"] = df.groupby("socket")["price"].rank(pct=True)
    elif cat == "motherboard":
        df["ddr"] = [ddr_placa(s, n) for s, n in zip(df["socket"], df["name"])]
        df["tam_placa"] = df["form_factor"].map(tamanho_placa)
    elif cat == "case":
        df["tam_max"] = df["type"].map(tamanho_max_gabinete)
    elif cat == "video-card":
        df["watts"] = df["chipset"].map(consumo_gpu)
    return df


def carregar_catalogo(somente_com_preco: bool = True) -> dict[str, pd.DataFrame]:
    return {c: carregar_categoria(c, somente_com_preco) for c in CATEGORIAS_MONTAGEM}


def carregar_catalogo_completo() -> pd.DataFrame:
    """Todas as 25 categorias (63 mil itens) — usado no benchmark de ANN."""
    partes = []
    for f in sorted(DADOS.glob("*.parquet")):
        if f.stem.endswith("_com_preco"):
            continue
        partes.append(pd.read_parquet(f))
    return pd.concat(partes, ignore_index=True)
