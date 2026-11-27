"""Testes de integração para validação das métricas globais e saídas do dataset."""

from pathlib import Path
import pandas as pd
from src.main import resolver_diretorio_input
from src.load import carregar_todas_categorias
from src.common import remover_duplicatas


def test_totais_globais_linhas_lidas_e_duplicatas():
    """Valida se o total lido é exatamente 66.778 e duplicatas removidas são 2.951."""
    dir_input = resolver_diretorio_input("data/raw/data/json")
    dfs = carregar_todas_categorias(dir_input)

    total_lidas = 0
    total_duplicatas = 0

    for cat, df in dfs.items():
        total_lidas += len(df)
        _, removidas = remover_duplicatas(df)
        total_duplicatas += removidas

    assert total_lidas == 66778, f"Esperado 66.778 linhas lidas, obtido: {total_lidas}"
    assert total_duplicatas == 2951, f"Esperado 2.951 duplicatas removidas, obtido: {total_duplicatas}"
    assert (total_lidas - total_duplicatas) == 63827


def test_resumo_limpeza_csv_gerado_e_consistente():
    """Valida se o arquivo resumo_limpeza.csv existe e se a soma das colunas bate com os totais."""
    caminho_resumo = Path("data/processed/resumo_limpeza.csv")
    if not caminho_resumo.exists():
        caminho_resumo = Path("pc_parts_cleaning/data/processed/resumo_limpeza.csv")

    assert caminho_resumo.exists(), "resumo_limpeza.csv não foi encontrado!"

    df_resumo = pd.read_csv(caminho_resumo)
    assert len(df_resumo) == 25
    assert df_resumo["linhas_lidas"].sum() == 66778
    assert df_resumo["duplicatas_removidas"].sum() == 2951
    assert df_resumo["linhas_finais"].sum() == 63827

