"""Módulo para carregamento dos arquivos JSON brutos do PC Part Dataset."""

import json
from pathlib import Path
import pandas as pd


def carregar_categoria(caminho_arquivo: Path | str) -> pd.DataFrame:
    """Lê um arquivo JSON de categoria e adiciona a coluna 'categoria'."""
    caminho = Path(caminho_arquivo)
    with open(caminho, mode="r", encoding="utf-8") as arquivo:
        dados = json.load(arquivo)

    df = pd.DataFrame(dados)
    categoria = caminho.stem
    df["categoria"] = categoria
    return df


def carregar_todas_categorias(diretorio_json: Path | str) -> dict[str, pd.DataFrame]:
    """Carrega todos os arquivos JSON de categorias em um diretório."""
    diretorio = Path(diretorio_json)
    arquivos = sorted(diretorio.glob("*.json"))
    dataframes: dict[str, pd.DataFrame] = {}

    for arquivo in arquivos:
        df = carregar_categoria(arquivo)
        dataframes[arquivo.stem] = df

    return dataframes

