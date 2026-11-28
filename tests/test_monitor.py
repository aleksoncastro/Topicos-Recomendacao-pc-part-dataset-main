"""Testes específicos para a categoria monitor."""

from pathlib import Path
import pandas as pd
from src.limpeza.main import resolver_diretorio_input
from src.limpeza.load import carregar_categoria
from src.limpeza.common import remover_duplicatas, gerar_id, extrair_marca, adicionar_tem_preco, calcular_outlier_preco, normalizar_numericos
from src.limpeza.categories import tratar_monitor


def test_monitor_resolucao_e_megapixels():
    """Valida separação de resolução em largura, altura e megapixels sem listas restantes."""
    dir_input = resolver_diretorio_input("data/raw")
    df_raw = carregar_categoria(dir_input / "monitor.json")
    df_dedup, _ = remover_duplicatas(df_raw)
    df_id = gerar_id(df_dedup)
    df_marca = extrair_marca(df_id)
    df_cat = tratar_monitor(df_marca)
    df_preco = adicionar_tem_preco(df_cat)
    df_final = normalizar_numericos(calcular_outlier_preco(df_preco))

    assert "resolution" not in df_final.columns
    assert "res_width" in df_final.columns
    assert "res_height" in df_final.columns
    assert "megapixels" in df_final.columns

    # Validar cálculo de megapixels
    esperado = (df_final["res_width"] * df_final["res_height"]) / 1_000_000.0
    pd.testing.assert_series_equal(df_final["megapixels"], esperado, check_names=False)

    # Nenhuma lista restante
    for col in df_final.columns:
        tem_lista = df_final[col].apply(lambda x: isinstance(x, (list, tuple))).any()
        assert not tem_lista, f"Coluna '{col}' ainda contém listas em monitor!"
