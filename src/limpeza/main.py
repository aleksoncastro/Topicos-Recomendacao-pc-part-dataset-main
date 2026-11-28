"""Ponto de entrada (CLI) para execução do pipeline de limpeza e pré-processamento."""

import argparse
from pathlib import Path
import pandas as pd

from src.load import carregar_categoria
from src.common import (
    remover_duplicatas,
    gerar_id,
    extrair_marca,
    adicionar_tem_preco,
    calcular_outlier_preco,
    normalizar_numericos,
)
from src.categories import aplicar_regras_categoria
from src.report import (
    gerar_resumo_limpeza,
    salvar_resumo,
    identificar_marcas_raras,
    log_progresso,
)


def resolver_diretorio_input(caminho_input: Path | str) -> Path:
    """Resolve o diretório correto contendo os arquivos .json da base."""
    caminho = Path(caminho_input)
    if caminho.exists() and list(caminho.glob("*.json")):
        return caminho

    # Tenta subdiretórios comuns caso o usuário passe apenas a raiz
    candidatos = [
        caminho / "data" / "json",
        caminho / "json",
        Path("pc_parts_cleaning") / caminho,
        Path("pc_parts_cleaning") / "data" / "raw" / "data" / "json",
    ]
    for cand in candidatos:
        if cand.exists() and list(cand.glob("*.json")):
            return cand

    return caminho


def processar_categoria(
    caminho_arquivo: Path, diretorio_saida: Path
) -> tuple[dict, pd.DataFrame]:
    """Executa o pipeline completo de limpeza para uma categoria e salva os arquivos."""
    categoria = caminho_arquivo.stem
    log_progresso(categoria, "Iniciando processamento")

    # 1. Leitura
    df_raw = carregar_categoria(caminho_arquivo)
    linhas_lidas = len(df_raw)

    # 2. Remoção de duplicatas
    df_limpo, duplicatas = remover_duplicatas(df_raw)

    # 3. Geração de ID hash
    df_com_id = gerar_id(df_limpo)

    # 4. Extração de marca
    df_marca = extrair_marca(df_com_id)

    # Regras específicas por categoria
    df_cat = aplicar_regras_categoria(df_marca, categoria)

    # 5. Coluna tem_preco
    df_preco = adicionar_tem_preco(df_cat)

    # 6. Outliers de preço
    df_outlier = calcular_outlier_preco(df_preco)

    # 7. Normalização de numéricos
    df_final = normalizar_numericos(df_outlier)

    linhas_finais = len(df_final)
    linhas_com_preco = int(df_final["tem_preco"].sum())
    n_outliers = int(df_final["outlier_preco"].sum())
    pct_com_preco = (
        round((linhas_com_preco / linhas_finais) * 100.0, 2)
        if linhas_finais > 0
        else 0.0
    )

    # Salvar saídas
    diretorio_saida.mkdir(parents=True, exist_ok=True)
    caminho_parquet = diretorio_saida / f"{categoria}.parquet"
    caminho_csv = diretorio_saida / f"{categoria}.csv"
    caminho_com_preco = diretorio_saida / f"{categoria}_com_preco.parquet"

    df_final.to_parquet(caminho_parquet, index=False)
    df_final.to_csv(caminho_csv, index=False, encoding="utf-8")

    df_com_preco = df_final[df_final["tem_preco"]].copy()
    df_com_preco.to_parquet(caminho_com_preco, index=False)

    log_progresso(
        categoria,
        "Concluído",
        f"Lidas: {linhas_lidas} | Dedup: -{duplicatas} | Finais: {linhas_finais} | Com Preço: {pct_com_preco}% | Outliers: {n_outliers}",
    )

    metricas = {
        "categoria": categoria,
        "linhas_lidas": linhas_lidas,
        "duplicatas_removidas": duplicatas,
        "linhas_finais": linhas_finais,
        "pct_com_preco": pct_com_preco,
        "outliers_preco": n_outliers,
    }

    return metricas, df_final


def executar_pipeline(caminho_input: str, caminho_output: str) -> None:
    """Executa o pipeline em lote para todas as categorias encontradas."""
    dir_input = resolver_diretorio_input(caminho_input)
    dir_output = Path(caminho_output)

    arquivos_json = sorted(dir_input.glob("*.json"))
    if not arquivos_json:
        raise FileNotFoundError(
            f"Nenhum arquivo .json encontrado no diretório: {dir_input}"
        )

    print(f"=== PC PART DATASET CLEANER ===")
    print(f"Diretório de entrada: {dir_input}")
    print(f"Diretório de saída:   {dir_output}")
    print(f"Total de categorias:  {len(arquivos_json)}\n")

    estatisticas: list[dict] = []
    dataframes_processados: list[pd.DataFrame] = []

    for arq in arquivos_json:
        metricas, df_proc = processar_categoria(arq, dir_output)
        estatisticas.append(metricas)
        dataframes_processados.append(df_proc)

    # Gerar e salvar resumo consolidado
    df_resumo = gerar_resumo_limpeza(estatisticas)
    caminho_resumo = dir_output / "resumo_limpeza.csv"
    salvar_resumo(df_resumo, caminho_resumo)

    total_lidas = sum(m["linhas_lidas"] for m in estatisticas)
    total_dedup = sum(m["duplicatas_removidas"] for m in estatisticas)
    total_finais = sum(m["linhas_finais"] for m in estatisticas)
    total_outliers = sum(m["outliers_preco"] for m in estatisticas)

    print("\n=== RESUMO GERAL ===")
    print(f"Total de linhas lidas:          {total_lidas:,}".replace(",", "."))
    print(f"Total de duplicatas removidas:  {total_dedup:,}".replace(",", "."))
    print(f"Total de registros finais:      {total_finais:,}".replace(",", "."))
    print(f"Total de outliers de preço:     {total_outliers:,}".replace(",", "."))
    print(f"Arquivo de resumo gerado em:    {caminho_resumo}\n")

    # Listar marcas com menos de 3 ocorrências
    marcas_raras = identificar_marcas_raras(dataframes_processados, limite_ocorrencias=3)
    print(f"=== MARCAS COM MENOS DE 3 OCORRÊNCIAS ({len(marcas_raras)} marcas encontradas) ===")
    for marca, contagem in marcas_raras:
        print(f"  - {marca}: {contagem} ocorrência(s)")


def main() -> None:
    """Função principal com parsing de argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Limpeza e pré-processamento do PC Part Dataset."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/data/json",
        help="Caminho para o diretório contendo os arquivos .json originais.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Caminho para o diretório de destino dos arquivos processados.",
    )

    args = parser.parse_args()
    executar_pipeline(args.input, args.output)


if __name__ == "__main__":
    main()

