"""Módulo de etapas comuns aplicadas a todas as categorias de peças."""

import hashlib
import json
import pandas as pd


# Dicionário de exceções para marcas compostas (chave em minúsculas -> forma canônica)
EXCECOES_MARCAS: dict[str, str] = {
    "western digital": "Western Digital",
    "cooler master": "Cooler Master",
    "thermal take": "Thermal Take",
    "thermaltake": "Thermaltake",
    "be quiet!": "be quiet!",
    "lian li": "Lian Li",
    "g.skill": "G.Skill",
    "silicon power": "Silicon Power",
    "fractal design": "Fractal Design",
    "in win": "In Win",
    "tripp lite": "Tripp Lite",
    "creative labs": "Creative Labs",
    "fsp group": "FSP Group",
    "super flower": "Super Flower",
    "arctic silver": "Arctic Silver",
    "visiontek": "VisionTek",
    "powercolor": "PowerColor",
}


def remover_duplicatas(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove duplicatas exatas convertendo listas em tuplas para hashing."""
    df_temp = df.copy()
    for col in df_temp.columns:
        if df_temp[col].apply(lambda x: isinstance(x, (list, tuple))).any():
            df_temp[col] = df_temp[col].apply(lambda x: tuple(x) if isinstance(x, (list, tuple)) else x)

    linhas_antes = len(df_temp)
    df_limpo = df_temp.drop_duplicates().copy()
    duplicatas_removidas = linhas_antes - len(df_limpo)
    return df_limpo, duplicatas_removidas


def gerar_id(df: pd.DataFrame) -> pd.DataFrame:
    """Cria a coluna 'id' como hash SHA1 da serialização de todos os campos da linha."""
    df_res = df.copy()

    def serializar_e_hashear(registro: dict) -> str:
        payload = json.dumps(registro, sort_keys=True, ensure_ascii=True, default=str)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    registros = df_res.to_dict(orient="records")
    ids = [serializar_e_hashear(r) for r in registros]
    df_res.insert(0, "id", ids)
    return df_res


def _extrair_marca_de_nome(nome: str | None) -> str | None:
    """Extrai a marca a partir do nome utilizando lista de exceções e primeira palavra."""
    if not isinstance(nome, str) or not nome.strip():
        return None

    nome_limpo = nome.strip()
    nome_lower = nome_limpo.lower()

    # Ordenar por tamanho decrescente da chave para casar prefixos mais longos primeiro
    for chave, canonico in sorted(EXCECOES_MARCAS.items(), key=lambda item: len(item[0]), reverse=True):
        if nome_lower == chave or nome_lower.startswith(chave + " "):
            return canonico

    # Caso padrão: primeira palavra
    return nome_limpo.split()[0]


def extrair_marca(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona a coluna 'marca' extraída do 'name' mantendo o 'name' original."""
    df_res = df.copy()
    marcas = df_res["name"].apply(_extrair_marca_de_nome)
    
    # Inserir coluna marca logo após 'name' se existir, ou no final
    if "name" in df_res.columns:
        pos = df_res.columns.get_loc("name") + 1
        df_res.insert(pos, "marca", marcas)
    else:
        df_res["marca"] = marcas

    return df_res


def adicionar_tem_preco(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona a coluna booleana 'tem_preco' indicando se price não é nulo."""
    df_res = df.copy()
    df_res["tem_preco"] = df_res["price"].notna().astype(bool)
    return df_res


def calcular_outlier_preco(df: pd.DataFrame) -> pd.DataFrame:
    """Sinaliza outliers de preço usando a regra de 1.5*IQR calculada sobre linhas com preço."""
    df_res = df.copy()
    precos = df_res.loc[df_res["tem_preco"], "price"].dropna()

    if len(precos) > 0:
        q1 = float(precos.quantile(0.25))
        q3 = float(precos.quantile(0.75))
        iqr = q3 - q1
        limite_inf = q1 - 1.5 * iqr
        limite_sup = q3 + 1.5 * iqr

        is_outlier = df_res["tem_preco"] & (
            (df_res["price"] < limite_inf) | (df_res["price"] > limite_sup)
        )
        df_res["outlier_preco"] = is_outlier.astype(bool)
    else:
        df_res["outlier_preco"] = False

    return df_res


def normalizar_numericos(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza colunas numéricas (inteiros e floats) para float64."""
    df_res = df.copy()
    for col in df_res.columns:
        # Se a coluna já é numérica e não booleana, converte para float64
        if pd.api.types.is_numeric_dtype(df_res[col]) and not pd.api.types.is_bool_dtype(df_res[col]):
            df_res[col] = df_res[col].astype("float64")
    return df_res

