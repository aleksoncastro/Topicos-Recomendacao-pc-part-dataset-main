"""Testes para as categorias cpu-cooler e case-fan."""

from pathlib import Path
from src.limpeza.main import resolver_diretorio_input
from src.limpeza.load import carregar_categoria
from src.limpeza.common import remover_duplicatas, gerar_id, extrair_marca, adicionar_tem_preco, calcular_outlier_preco, normalizar_numericos
from src.limpeza.categories import tratar_cpu_cooler, tratar_case_fan


def test_cpu_cooler_sem_listas_e_faixas_min_max():
    """Valida ausência de listas e criação de min/max em cpu-cooler."""
    dir_input = resolver_diretorio_input("data/raw")
    df_raw = carregar_categoria(dir_input / "cpu-cooler.json")
    df_dedup, _ = remover_duplicatas(df_raw)
    df_id = gerar_id(df_dedup)
    df_marca = extrair_marca(df_id)
    df_cat = tratar_cpu_cooler(df_marca)
    df_preco = adicionar_tem_preco(df_cat)
    df_final = normalizar_numericos(calcular_outlier_preco(df_preco))

    # Colunas originais não devem existir
    assert "rpm" not in df_final.columns
    assert "noise_level" not in df_final.columns

    # Colunas min e max criadas
    assert "rpm_min" in df_final.columns
    assert "rpm_max" in df_final.columns
    assert "noise_level_min" in df_final.columns
    assert "noise_level_max" in df_final.columns

    # Nenhuma lista restante
    for col in df_final.columns:
        tem_lista = df_final[col].apply(lambda x: isinstance(x, (list, tuple))).any()
        assert not tem_lista, f"Coluna '{col}' ainda contém listas em cpu-cooler!"


def test_case_fan_sem_listas_e_faixas_min_max():
    """Valida ausência de listas e criação de min/max em case-fan."""
    dir_input = resolver_diretorio_input("data/raw")
    df_raw = carregar_categoria(dir_input / "case-fan.json")
    df_dedup, _ = remover_duplicatas(df_raw)
    df_id = gerar_id(df_dedup)
    df_marca = extrair_marca(df_id)
    df_cat = tratar_case_fan(df_marca)
    df_preco = adicionar_tem_preco(df_cat)
    df_final = normalizar_numericos(calcular_outlier_preco(df_preco))

    # Colunas originais não devem existir
    assert "rpm" not in df_final.columns
    assert "airflow" not in df_final.columns
    assert "noise_level" not in df_final.columns

    # Colunas min e max criadas
    assert "rpm_min" in df_final.columns
    assert "rpm_max" in df_final.columns
    assert "airflow_min" in df_final.columns
    assert "airflow_max" in df_final.columns
    assert "noise_level_min" in df_final.columns
    assert "noise_level_max" in df_final.columns

    # Nenhuma lista restante
    for col in df_final.columns:
        tem_lista = df_final[col].apply(lambda x: isinstance(x, (list, tuple))).any()
        assert not tem_lista, f"Coluna '{col}' ainda contém listas em case-fan!"

