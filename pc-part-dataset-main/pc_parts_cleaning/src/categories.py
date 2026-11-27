"""Módulo contendo regras de transformação específicas por categoria."""

import numpy as np
import pandas as pd


def _extrair_min_max(valor: object) -> tuple[float, float]:
    """Extrai valores mínimo e máximo a partir de número único ou lista/tupla [min, max]."""
    if isinstance(valor, (list, tuple)) and len(valor) >= 2:
        try:
            return float(valor[0]), float(valor[1])
        except (ValueError, TypeError):
            return np.nan, np.nan
    if isinstance(valor, (int, float)) and not pd.isna(valor):
        v = float(valor)
        return v, v
    return np.nan, np.nan


def tratar_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica regras específicas da categoria memory."""
    df_res = df.copy()

    # speed -> ddr_version e speed_mhz
    ddr_versoes: list[float] = []
    velocidades_mhz: list[float] = []
    for val in df_res["speed"]:
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            ddr_versoes.append(float(val[0]))
            velocidades_mhz.append(float(val[1]))
        elif val is not None and not pd.isna(val):
            ddr_versoes.append(np.nan)
            velocidades_mhz.append(float(val))
        else:
            ddr_versoes.append(np.nan)
            velocidades_mhz.append(np.nan)

    df_res["ddr_version"] = ddr_versoes
    df_res["speed_mhz"] = velocidades_mhz

    # modules -> n_modules e gb_per_module
    n_modulos: list[float] = []
    gb_por_modulo: list[float] = []
    for val in df_res["modules"]:
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            n_modulos.append(float(val[0]))
            gb_por_modulo.append(float(val[1]))
        else:
            n_modulos.append(np.nan)
            gb_por_modulo.append(np.nan)

    df_res["n_modules"] = n_modulos
    df_res["gb_per_module"] = gb_por_modulo
    df_res["total_gb"] = df_res["n_modules"] * df_res["gb_per_module"]

    # Remover colunas originais de lista
    df_res = df_res.drop(columns=["speed", "modules"])

    # Validação price_per_gb ≈ price / total_gb (tolerância 0.01)
    tem_dados = (
        df_res["price"].notna()
        & df_res["price_per_gb"].notna()
        & df_res["total_gb"].notna()
        & (df_res["total_gb"] > 0)
    )
    calculado = df_res["price"] / df_res["total_gb"]
    df_res["ppg_confere"] = (
        tem_dados & ((df_res["price_per_gb"] - calculado).abs() <= 0.01)
    ).astype(bool)

    return df_res


def tratar_internal_hard_drive(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica regras da categoria internal-hard-drive."""
    df_res = df.copy()

    drive_tipos: list[str] = []
    rpms: list[float] = []
    for t in df_res["type"]:
        if pd.isna(t) or t is None:
            drive_tipos.append("Desconhecido")
            rpms.append(np.nan)
        elif str(t).strip().upper() == "SSD":
            drive_tipos.append("SSD")
            rpms.append(np.nan)
        else:
            drive_tipos.append("HDD")
            try:
                rpms.append(float(t))
            except (ValueError, TypeError):
                rpms.append(np.nan)

    df_res["drive_type"] = drive_tipos
    df_res["rpm"] = rpms

    # Normalizar form_factor
    form_factors_norm: list[str | None] = []
    for ff in df_res["form_factor"]:
        if pd.isna(ff) or ff is None:
            form_factors_norm.append(None)
        elif isinstance(ff, (int, float)):
            form_factors_norm.append(f"{ff:g}in")
        elif str(ff).strip() in ["2.5", "3.5"]:
            form_factors_norm.append(f"{str(ff).strip()}in")
        else:
            form_factors_norm.append(str(ff).strip())

    df_res["form_factor_norm"] = form_factors_norm
    df_res = df_res.drop(columns=["type", "form_factor"])

    return df_res


def tratar_power_supply(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica regras da categoria power-supply."""
    df_res = df.copy()

    # modular: False -> 'Nao modular', mantendo strings
    modulares: list[str | None] = []
    for m in df_res["modular"]:
        if m is False or str(m).strip().lower() == "false":
            modulares.append("Nao modular")
        elif pd.isna(m) or m is None:
            modulares.append(None)
        else:
            modulares.append(str(m).strip())

    df_res["modular"] = modulares
    return df_res


def tratar_cpu_cooler(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica regras da categoria cpu-cooler."""
    df_res = df.copy()

    rpm_mins, rpm_maxs = [], []
    for val in df_res["rpm"]:
        vmin, vmax = _extrair_min_max(val)
        rpm_mins.append(vmin)
        rpm_maxs.append(vmax)
    df_res["rpm_min"] = rpm_mins
    df_res["rpm_max"] = rpm_maxs

    noise_mins, noise_maxs = [], []
    for val in df_res["noise_level"]:
        vmin, vmax = _extrair_min_max(val)
        noise_mins.append(vmin)
        noise_maxs.append(vmax)
    df_res["noise_level_min"] = noise_mins
    df_res["noise_level_max"] = noise_maxs

    df_res = df_res.drop(columns=["rpm", "noise_level"])
    return df_res


def tratar_case_fan(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica regras da categoria case-fan."""
    df_res = df.copy()

    rpm_mins, rpm_maxs = [], []
    for val in df_res["rpm"]:
        vmin, vmax = _extrair_min_max(val)
        rpm_mins.append(vmin)
        rpm_maxs.append(vmax)
    df_res["rpm_min"] = rpm_mins
    df_res["rpm_max"] = rpm_maxs

    airflow_mins, airflow_maxs = [], []
    for val in df_res["airflow"]:
        vmin, vmax = _extrair_min_max(val)
        airflow_mins.append(vmin)
        airflow_maxs.append(vmax)
    df_res["airflow_min"] = airflow_mins
    df_res["airflow_max"] = airflow_maxs

    noise_mins, noise_maxs = [], []
    for val in df_res["noise_level"]:
        vmin, vmax = _extrair_min_max(val)
        noise_mins.append(vmin)
        noise_maxs.append(vmax)
    df_res["noise_level_min"] = noise_mins
    df_res["noise_level_max"] = noise_maxs

    df_res = df_res.drop(columns=["rpm", "airflow", "noise_level"])
    return df_res


def tratar_monitor(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica regras da categoria monitor."""
    df_res = df.copy()

    larguras: list[float] = []
    alturas: list[float] = []
    for val in df_res["resolution"]:
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            larguras.append(float(val[0]))
            alturas.append(float(val[1]))
        else:
            larguras.append(np.nan)
            alturas.append(np.nan)

    df_res["res_width"] = larguras
    df_res["res_height"] = alturas
    df_res["megapixels"] = (df_res["res_width"] * df_res["res_height"]) / 1_000_000.0
    df_res = df_res.drop(columns=["resolution"])
    return df_res


def tratar_cpu(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica regras da categoria cpu."""
    df_res = df.copy()
    df_res["gpu_integrada"] = df_res["graphics"].notna().astype(bool)
    return df_res


def tratar_headphones(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica regras da categoria headphones."""
    df_res = df.copy()

    freq_mins, freq_maxs = [], []
    for val in df_res["frequency_response"]:
        vmin, vmax = _extrair_min_max(val)
        freq_mins.append(vmin)
        freq_maxs.append(vmax)
    df_res["freq_resp_min"] = freq_mins
    df_res["freq_resp_max"] = freq_maxs
    df_res = df_res.drop(columns=["frequency_response"])
    return df_res


def tratar_speakers(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica regras da categoria speakers."""
    df_res = df.copy()

    freq_mins, freq_maxs = [], []
    for val in df_res["frequency_response"]:
        vmin, vmax = _extrair_min_max(val)
        freq_mins.append(vmin)
        freq_maxs.append(vmax)
    df_res["freq_resp_min"] = freq_mins
    df_res["freq_resp_max"] = freq_maxs
    df_res = df_res.drop(columns=["frequency_response"])
    return df_res


def tratar_webcam(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica regras da categoria webcam."""
    df_res = df.copy()

    def formatar_lista(val: object) -> str | None:
        if isinstance(val, (list, tuple)):
            return ", ".join(str(item) for item in val if item is not None)
        if val is not None and not pd.isna(val):
            return str(val)
        return None

    df_res["resolutions"] = df_res["resolutions"].apply(formatar_lista)
    df_res["os"] = df_res["os"].apply(formatar_lista)
    return df_res


def tratar_outras_com_listas(df: pd.DataFrame, categoria: str) -> pd.DataFrame:
    """Trata categorias secundárias que contêm listas para garantir compatibilidade com Parquet."""
    df_res = df.copy()
    if categoria == "os" and "mode" in df_res.columns:
        df_res["mode"] = df_res["mode"].apply(
            lambda x: "/".join(str(i) for i in x) if isinstance(x, (list, tuple)) else (str(x) if pd.notna(x) else None)
        )
    elif categoria == "fan-controller" and "form_factor" in df_res.columns:
        df_res["form_factor"] = df_res["form_factor"].apply(
            lambda x: f"{x[0]}x {x[1]}in" if isinstance(x, (list, tuple)) else (f"{x:g}in" if isinstance(x, (int, float)) else str(x) if pd.notna(x) else None)
        )
    return df_res


MAPEAMENTO_CATEGORIAS = {
    "memory": tratar_memory,
    "internal-hard-drive": tratar_internal_hard_drive,
    "power-supply": tratar_power_supply,
    "cpu-cooler": tratar_cpu_cooler,
    "case-fan": tratar_case_fan,
    "monitor": tratar_monitor,
    "cpu": tratar_cpu,
    "headphones": tratar_headphones,
    "speakers": tratar_speakers,
    "webcam": tratar_webcam,
}


def aplicar_regras_categoria(df: pd.DataFrame, categoria: str) -> pd.DataFrame:
    """Aplica a regra específica correspondente à categoria, se existir."""
    if categoria in MAPEAMENTO_CATEGORIAS:
        df = MAPEAMENTO_CATEGORIAS[categoria](df)
    df = tratar_outras_com_listas(df, categoria)
    return df

