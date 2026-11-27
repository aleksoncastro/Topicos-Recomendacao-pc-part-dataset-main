"""Testes específicos para o pipeline da categoria memory."""

import numpy as np
import pandas as pd
from pathlib import Path
from src.load import carregar_categoria
from src.common import (
    remover_duplicatas,
    gerar_id,
    extrair_marca,
    adicionar_tem_preco,
    calcular_outlier_preco,
    normalizar_numericos,
)
from src.categories import tratar_memory


CAMINHO_RAW = Path("data/raw/data/json")
if not CAMINHO_RAW.exists():
    CAMINHO_RAW = Path("pc_parts_cleaning/data/raw/data/json")


def processar_memory_completo():
    """Executa o pipeline completo para a categoria memory."""
    caminho_memory = CAMINHO_RAW / "memory.json"
    df_raw = carregar_categoria(caminho_memory)
    total_lidas = len(df_raw)

    df_dedup, duplicatas = remover_duplicatas(df_raw)
    df_id = gerar_id(df_dedup)
    df_marca = extrair_marca(df_id)
    df_cat = tratar_memory(df_marca)
    df_preco = adicionar_tem_preco(df_cat)
    df_outlier = calcular_outlier_preco(df_preco)
    df_final = normalizar_numericos(df_outlier)

    return df_raw, df_final, total_lidas, duplicatas


def test_memory_totais_e_deduplicacao():
    """Valida total de linhas lidas e duplicatas removidas em memory."""
    _, df_final, total_lidas, duplicatas = processar_memory_completo()
    assert total_lidas == 13553
    assert duplicatas == 957
    assert len(df_final) == 12596


def test_memory_ids_unicos():
    """Valida se todos os IDs gerados para memory são únicos."""
    _, df_final, _, _ = processar_memory_completo()
    assert df_final["id"].nunique() == len(df_final)


def test_memory_quatro_registros_speed_inteiro():
    """Valida se os 4 registros com speed inteiro têm ddr_version=NaN e speed_mhz preenchido."""
    _, df_final, _, _ = processar_memory_completo()
    nans_ddr = df_final["ddr_version"].isna().sum()
    assert nans_ddr == 4

    # Verifica os 4 registros sem ddr_version
    linhas_sem_ddr = df_final[df_final["ddr_version"].isna()]
    assert set(linhas_sem_ddr["speed_mhz"].tolist()) == {333.0, 400.0}
    assert (linhas_sem_ddr["speed_mhz"].notna()).all()


def test_memory_sem_listas_restantes():
    """Valida que nenhuma coluna de memory contém listas ou tuplas após o tratamento."""
    _, df_final, _, _ = processar_memory_completo()
    for col in df_final.columns:
        tem_lista = df_final[col].apply(lambda x: isinstance(x, (list, tuple))).any()
        assert not tem_lista, f"Coluna '{col}' ainda contém listas ou tuplas!"


def test_memory_total_gb_e_ppg_confere():
    """Valida cálculo de total_gb e coluna ppg_confere."""
    _, df_final, _, _ = processar_memory_completo()
    assert "total_gb" in df_final.columns
    assert "ppg_confere" in df_final.columns

    # total_gb = n_modules * gb_per_module
    calculado = df_final["n_modules"] * df_final["gb_per_module"]
    np.testing.assert_allclose(df_final["total_gb"], calculado)

    # ppg_confere é booleana
    assert df_final["ppg_confere"].dtype == bool
    assert df_final["ppg_confere"].sum() > 0


def test_memory_precos_nao_imputados():
    """Garante que a contagem de preços não nulos não aumentou."""
    df_raw, df_final, _, _ = processar_memory_completo()
    assert df_final["tem_preco"].sum() <= df_raw["price"].notna().sum()
    assert df_final["tem_preco"].sum() == df_final["price"].notna().sum()

