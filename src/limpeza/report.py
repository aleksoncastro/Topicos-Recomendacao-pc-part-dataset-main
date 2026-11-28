"""Módulo para geração de relatórios de auditoria, métricas e logging."""

from collections import Counter
from pathlib import Path
import pandas as pd


def gerar_resumo_limpeza(estatisticas: list[dict]) -> pd.DataFrame:
    """Gera DataFrame de resumo com métricas de limpeza por categoria."""
    df_resumo = pd.DataFrame(estatisticas)
    colunas_ordenadas = [
        "categoria",
        "linhas_lidas",
        "duplicatas_removidas",
        "linhas_finais",
        "pct_com_preco",
        "outliers_preco",
    ]
    return df_resumo[colunas_ordenadas]


def salvar_resumo(df_resumo: pd.DataFrame, caminho_saida: Path | str) -> None:
    """Salva o resumo consolidado de limpeza em formato CSV."""
    caminho = Path(caminho_saida)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df_resumo.to_csv(caminho, index=False, encoding="utf-8")


def identificar_marcas_raras(
    dataframes: list[pd.DataFrame], limite_ocorrencias: int = 3
) -> list[tuple[str, int]]:
    """Identifica marcas distintas com contagem de ocorrências inferior ao limite."""
    contador: Counter[str] = Counter()
    for df in dataframes:
        if "marca" in df.columns:
            marcas = df["marca"].dropna().astype(str).tolist()
            contador.update(marcas)

    marcas_raras = [
        (marca, contagem)
        for marca, contagem in contador.items()
        if contagem < limite_ocorrencias
    ]
    # Ordenar por ocorrência crescente e depois ordem alfabética
    marcas_raras.sort(key=lambda item: (item[1], item[0]))
    return marcas_raras


def log_progresso(categoria: str, etapa: str, detalhes: str = "") -> None:
    """Exibe mensagem padronizada de progresso no terminal."""
    extra = f" - {detalhes}" if detalhes else ""
    print(f"[{categoria:<22}] {etapa:<25}{extra}")

