"""Testes para validação das regras comuns aplicadas a todas as categorias."""

from pathlib import Path
import pandas as pd
from src.limpeza.main import resolver_diretorio_input
from src.limpeza.load import carregar_todas_categorias
from src.limpeza.common import (
    remover_duplicatas,
    gerar_id,
    extrair_marca,
    adicionar_tem_preco,
    calcular_outlier_preco,
)


def test_ids_unicos_dentro_de_cada_categoria():
    """Valida se a coluna 'id' gerada é estritamente única dentro de cada categoria."""
    dir_input = resolver_diretorio_input("data/raw")
    dfs = carregar_todas_categorias(dir_input)

    for cat, df in dfs.items():
        df_dedup, _ = remover_duplicatas(df)
        df_com_id = gerar_id(df_dedup)
        assert df_com_id["id"].nunique() == len(df_com_id), f"Colisão de ID na categoria {cat}!"


def test_nenhum_preco_imputado_em_todas_categorias():
    """Garante que a contagem de preços não nulos não aumenta após o pré-processamento."""
    dir_input = resolver_diretorio_input("data/raw")
    dfs = carregar_todas_categorias(dir_input)

    for cat, df in dfs.items():
        df_dedup, _ = remover_duplicatas(df)
        df_preco = adicionar_tem_preco(df_dedup)

        precos_originais = df["price"].notna().sum()
        precos_dedup = df_dedup["price"].notna().sum()
        precos_sinalizados = df_preco["tem_preco"].sum()

        assert precos_sinalizados <= precos_originais
        assert precos_sinalizados == precos_dedup


def test_extrair_marca_compostas_e_simples():
    """Valida a extração de marcas simples e marcas compostas com exceções."""
    dados = {
        "name": [
            "Western Digital Blue 1 TB",
            "Cooler Master Hyper 212",
            "G.Skill Ripjaws V 16 GB",
            "be quiet! Pure Rock 2",
            "Lian Li O11 Dynamic",
            "Corsair Vengeance LPX 16 GB",
            "Samsung 970 Evo 1 TB",
            "Thermal Take Toughpower 750W",
            "Silicon Power A55 512 GB",
        ]
    }
    df = pd.DataFrame(dados)
    df_res = extrair_marca(df)

    marcas_esperadas = [
        "Western Digital",
        "Cooler Master",
        "G.Skill",
        "be quiet!",
        "Lian Li",
        "Corsair",
        "Samsung",
        "Thermal Take",
        "Silicon Power",
    ]
    assert df_res["marca"].tolist() == marcas_esperadas


def test_calcular_outliers_preco_iqr():
    """Valida identificação de outliers através da regra de 1.5*IQR."""
    # Conjunto sintético com outlier superior claro
    df = pd.DataFrame({
        "price": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 100.0, None],
        "tem_preco": [True, True, True, True, True, True, True, False],
    })
    df_out = calcular_outlier_preco(df)
    assert df_out.loc[df_out["price"] == 100.0, "outlier_preco"].iloc[0] == True
    assert df_out.loc[df_out["price"] == 12.0, "outlier_preco"].iloc[0] == False
    assert df_out.loc[df_out["price"].isna(), "outlier_preco"].iloc[0] == False

