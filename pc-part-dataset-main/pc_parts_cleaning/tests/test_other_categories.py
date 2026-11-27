"""Testes para regras de categorias adicionais (cpu, power-supply, headphones, speakers, webcam)."""

from pathlib import Path
from src.main import resolver_diretorio_input
from src.load import carregar_categoria
from src.common import remover_duplicatas, gerar_id, extrair_marca, adicionar_tem_preco, calcular_outlier_preco, normalizar_numericos
from src.categories import (
    tratar_cpu,
    tratar_power_supply,
    tratar_headphones,
    tratar_speakers,
    tratar_webcam,
)


def test_cpu_gpu_integrada():
    """Valida coluna booleana gpu_integrada na categoria cpu."""
    dir_input = resolver_diretorio_input("data/raw/data/json")
    df_raw = carregar_categoria(dir_input / "cpu.json")
    df_dedup, _ = remover_duplicatas(df_raw)
    df_final = tratar_cpu(df_dedup)

    assert "gpu_integrada" in df_final.columns
    assert df_final["gpu_integrada"].dtype == bool
    # Onde graphics não for nulo, deve ser True
    assert (df_final.loc[df_final["graphics"].notna(), "gpu_integrada"] == True).all()
    assert (df_final.loc[df_final["graphics"].isna(), "gpu_integrada"] == False).all()


def test_power_supply_modular():
    """Valida padronização de False para 'Nao modular' e preservação de nulos em efficiency."""
    dir_input = resolver_diretorio_input("data/raw/data/json")
    df_raw = carregar_categoria(dir_input / "power-supply.json")
    df_dedup, _ = remover_duplicatas(df_raw)
    df_final = tratar_power_supply(df_dedup)

    # Não deve existir booleano False em modular
    assert False not in df_final["modular"].values
    assert "Nao modular" in df_final["modular"].values

    # efficiency nulo permanece nulo
    orig_nulls = df_dedup["efficiency"].isna().sum()
    final_nulls = df_final["efficiency"].isna().sum()
    assert orig_nulls == final_nulls


def test_headphones_e_speakers_freq_resp():
    """Valida separação de frequency_response em min e max nas categorias de áudio."""
    dir_input = resolver_diretorio_input("data/raw/data/json")

    for cat_nome, func in [("headphones", tratar_headphones), ("speakers", tratar_speakers)]:
        df_raw = carregar_categoria(dir_input / f"{cat_nome}.json")
        df_dedup, _ = remover_duplicatas(df_raw)
        df_final = func(df_dedup)

        assert "frequency_response" not in df_final.columns
        assert "freq_resp_min" in df_final.columns
        assert "freq_resp_max" in df_final.columns

        for col in df_final.columns:
            tem_lista = df_final[col].apply(lambda x: isinstance(x, (list, tuple))).any()
            assert not tem_lista, f"Coluna '{col}' contém listas em {cat_nome}!"


def test_webcam_listas_formatadas_para_string():
    """Valida conversão de resolutions e os em strings legíveis na categoria webcam."""
    dir_input = resolver_diretorio_input("data/raw/data/json")
    df_raw = carregar_categoria(dir_input / "webcam.json")
    df_dedup, _ = remover_duplicatas(df_raw)
    df_final = tratar_webcam(df_dedup)

    for col in ["resolutions", "os"]:
        assert df_final[col].apply(lambda x: isinstance(x, (list, tuple))).sum() == 0
        valores_nao_nulos = df_final[col].dropna()
        assert all(isinstance(v, str) for v in valores_nao_nulos)

