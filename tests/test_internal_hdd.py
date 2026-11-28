"""Testes específicos para a categoria internal-hard-drive."""

import pandas as pd
from pathlib import Path
from src.limpeza.main import resolver_diretorio_input
from src.limpeza.load import carregar_categoria
from src.limpeza.common import remover_duplicatas, gerar_id, extrair_marca, adicionar_tem_preco, calcular_outlier_preco, normalizar_numericos
from src.limpeza.categories import tratar_internal_hard_drive


def processar_ihd():
    dir_input = resolver_diretorio_input("data/raw")
    df_raw = carregar_categoria(dir_input / "internal-hard-drive.json")
    df_dedup, _ = remover_duplicatas(df_raw)
    df_id = gerar_id(df_dedup)
    df_marca = extrair_marca(df_id)
    df_cat = tratar_internal_hard_drive(df_marca)
    df_preco = adicionar_tem_preco(df_cat)
    df_out = calcular_outlier_preco(df_preco)
    df_final = normalizar_numericos(df_out)
    return df_final


def test_ihd_drive_type_e_rpm():
    """Valida separação de type em drive_type e rpm."""
    df = processar_ihd()

    # drive_type tem apenas SSD, HDD e Desconhecido
    tipos_validos = {"SSD", "HDD", "Desconhecido"}
    assert set(df["drive_type"].unique()).issubset(tipos_validos)

    # 30 registros com drive_type = 'Desconhecido'
    assert (df["drive_type"] == "Desconhecido").sum() == 30

    # SSD e Desconhecido têm rpm como NaN
    assert df.loc[df["drive_type"] == "SSD", "rpm"].isna().all()
    assert df.loc[df["drive_type"] == "Desconhecido", "rpm"].isna().all()

    # HDD tem rpm numérico preenchido
    assert df.loc[df["drive_type"] == "HDD", "rpm"].notna().all()


def test_ihd_nenhuma_coluna_mistura_texto_e_numero():
    """Garante que nenhuma coluna em internal-hard-drive mistura texto e número."""
    df = processar_ihd()

    assert "type" not in df.columns
    assert "form_factor" not in df.columns

    for col in df.columns:
        valores_nao_nulos = df[col].dropna()
        if len(valores_nao_nulos) == 0:
            continue

        tipos = set(type(x) for x in valores_nao_nulos)

        # Se houver números (int/float), não deve haver strings
        tem_numero = any(issubclass(t, (int, float)) for t in tipos)
        tem_string = any(issubclass(t, str) for t in tipos)
        assert not (tem_numero and tem_string), f"Coluna '{col}' mistura texto e número: {tipos}"


def test_ihd_form_factor_norm():
    """Valida que form_factor_norm contém apenas strings padronizadas."""
    df = processar_ihd()
    assert "form_factor_norm" in df.columns
    valores = df["form_factor_norm"].dropna().unique()
    for v in valores:
        assert isinstance(v, str), f"Valor não é string em form_factor_norm: {v}"

    # 2.5 e 3.5 devem aparecer normalizados com sufixo 'in'
    assert "2.5in" in valores
    assert "3.5in" in valores
    assert "M.2-2280" in valores

