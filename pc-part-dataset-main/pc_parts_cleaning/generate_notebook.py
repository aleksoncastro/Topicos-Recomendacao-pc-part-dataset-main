"""Script para gerar o notebook Jupyter consolidado pc_part_dataset_cleaning.ipynb."""

import json
from pathlib import Path


def criar_notebook() -> None:
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    def add_md(source: str) -> None:
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")],
        })

    def add_code(source: str) -> None:
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")],
        })

    # Cell 1: Introdução
    add_md("""# PC Part Dataset — Limpeza, Auditoria e Pré-processamento
Este notebook consolida todas as etapas de limpeza, padronização, engenharia de atributos e validação do **PC Part Dataset** (peças de computador extraídas do PCPartPicker, com preços em USD).

### Destaques do Dataset:
- **25 categorias** de hardware em arquivos JSON
- **66.778 registros lidos** no total
- **2.951 duplicatas exatas** removidas
- **63.827 registros finais** limpos e padronizados
- Exportação em **Parquet** e **CSV** (tabela completa e tabela com preço)
- Relatório de auditoria consolidado (`resumo_limpeza.csv`) e verificação de marcas raras""")

    # Cell 2: Setup
    add_md("""## 1. Configuração do Ambiente e Importações""")
    add_code("""import os
import json
import hashlib
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np

# Configuração de exibição do Pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Localização automática dos diretórios
def encontrar_caminhos():
    candidatos_input = [
        Path('data/raw/data/json'),
        Path('pc_parts_cleaning/data/raw/data/json'),
        Path('../data/raw/data/json'),
        Path('data/json'),
        Path('../data/json')
    ]
    dir_input = None
    for cand in candidatos_input:
        if cand.exists() and list(cand.glob('*.json')):
            dir_input = cand
            break
            
    if dir_input is None:
        raise FileNotFoundError('Diretório de dados brutos (.json) não encontrado!')
        
    dir_output = Path('data/processed')
    if not dir_output.parent.exists() and Path('pc_parts_cleaning').exists():
        dir_output = Path('pc_parts_cleaning/data/processed')
        
    dir_output.mkdir(parents=True, exist_ok=True)
    return dir_input, dir_output

DIR_INPUT, DIR_OUTPUT = encontrar_caminhos()
print(f'Diretório de entrada: {DIR_INPUT.resolve()}')
print(f'Diretório de saída:   {DIR_OUTPUT.resolve()}')""")

    # Cell 3: Load
    add_md("""## 2. Funções de Carregamento dos Dados Brutos""")
    add_code("""def carregar_categoria(caminho_arquivo: Path | str) -> pd.DataFrame:
    '''Lê um arquivo JSON de categoria e adiciona a coluna categoria.'''
    caminho = Path(caminho_arquivo)
    with open(caminho, mode='r', encoding='utf-8') as f:
        dados = json.load(f)
    df = pd.DataFrame(dados)
    df['categoria'] = caminho.stem
    return df

def carregar_todas_categorias(diretorio: Path | str) -> dict[str, pd.DataFrame]:
    '''Carrega todos os arquivos JSON de categorias em ordem alfabética.'''
    arquivos = sorted(Path(diretorio).glob('*.json'))
    return {arq.stem: carregar_categoria(arq) for arq in arquivos}

print(f'Total de arquivos JSON encontrados: {len(list(DIR_INPUT.glob("*.json")))}')""")

    # Cell 4: Common
    add_md("""## 3. Etapas Comuns a Todas as Categorias
1. **Deduplicação exata**: remoção de linhas idênticas em todos os campos, convertendo listas para tuplas para permitir comparação por valor.
2. **Identificador único (`id`)**: hash SHA-1 hexadecimal de 40 caracteres da serialização ordenada de todos os campos.
3. **Extração de `marca`**: primeira palavra do `name`, com dicionário de exceções canônicas para marcas compostas (ex.: Western Digital, Cooler Master, be quiet!, Lian Li, G.Skill, etc.).
4. **Preço ausente (`tem_preco`)**: booleana indicando presença de preço, sem imputação.
5. **Outliers de preço (`outlier_preco`)**: identificação via regra de $1,5 \\times IQR$ calculada sobre as linhas com preço.
6. **Normalização numérica**: conversão de colunas numéricas para `float64`.""")
    add_code("""EXCECOES_MARCAS: dict[str, str] = {
    'western digital': 'Western Digital',
    'cooler master': 'Cooler Master',
    'thermal take': 'Thermal Take',
    'thermaltake': 'Thermaltake',
    'be quiet!': 'be quiet!',
    'lian li': 'Lian Li',
    'g.skill': 'G.Skill',
    'silicon power': 'Silicon Power',
    'fractal design': 'Fractal Design',
    'in win': 'In Win',
    'tripp lite': 'Tripp Lite',
    'creative labs': 'Creative Labs',
    'fsp group': 'FSP Group',
    'super flower': 'Super Flower',
    'arctic silver': 'Arctic Silver',
    'visiontek': 'VisionTek',
    'powercolor': 'PowerColor',
}

def remover_duplicatas(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    '''Remove duplicatas exatas convertendo listas em tuplas para hashing.'''
    df_temp = df.copy()
    for col in df_temp.columns:
        if df_temp[col].apply(lambda x: isinstance(x, (list, tuple))).any():
            df_temp[col] = df_temp[col].apply(lambda x: tuple(x) if isinstance(x, (list, tuple)) else x)
    linhas_antes = len(df_temp)
    df_limpo = df_temp.drop_duplicates().copy()
    return df_limpo, linhas_antes - len(df_limpo)

def gerar_id(df: pd.DataFrame) -> pd.DataFrame:
    '''Cria coluna id como hash SHA-1 determinístico dos campos da linha.'''
    df_res = df.copy()
    def hash_linha(registro: dict) -> str:
        payload = json.dumps(registro, sort_keys=True, ensure_ascii=True, default=str)
        return hashlib.sha1(payload.encode('utf-8')).hexdigest()
    registros = df_res.to_dict(orient='records')
    df_res.insert(0, 'id', [hash_linha(r) for r in registros])
    return df_res

def _extrair_marca_de_nome(nome: str | None) -> str | None:
    '''Extrai marca a partir do nome utilizando lista de exceções e primeira palavra.'''
    if not isinstance(nome, str) or not nome.strip():
        return None
    nome_limpo = nome.strip()
    nome_lower = nome_limpo.lower()
    for chave, canonico in sorted(EXCECOES_MARCAS.items(), key=lambda item: len(item[0]), reverse=True):
        if nome_lower == chave or nome_lower.startswith(chave + ' '):
            return canonico
    return nome_limpo.split()[0]

def extrair_marca(df: pd.DataFrame) -> pd.DataFrame:
    '''Adiciona coluna marca extraída de name.'''
    df_res = df.copy()
    marcas = df_res['name'].apply(_extrair_marca_de_nome)
    if 'name' in df_res.columns:
        pos = df_res.columns.get_loc('name') + 1
        df_res.insert(pos, 'marca', marcas)
    else:
        df_res['marca'] = marcas
    return df_res

def adicionar_tem_preco(df: pd.DataFrame) -> pd.DataFrame:
    '''Adiciona coluna booleana tem_preco.'''
    df_res = df.copy()
    df_res['tem_preco'] = df_res['price'].notna().astype(bool)
    return df_res

def calcular_outlier_preco(df: pd.DataFrame) -> pd.DataFrame:
    '''Sinaliza outliers de preço usando a regra 1.5*IQR sobre linhas com preço.'''
    df_res = df.copy()
    precos = df_res.loc[df_res['tem_preco'], 'price'].dropna()
    if len(precos) > 0:
        q1 = float(precos.quantile(0.25))
        q3 = float(precos.quantile(0.75))
        iqr = q3 - q1
        limite_inf = q1 - 1.5 * iqr
        limite_sup = q3 + 1.5 * iqr
        is_outlier = df_res['tem_preco'] & ((df_res['price'] < limite_inf) | (df_res['price'] > limite_sup))
        df_res['outlier_preco'] = is_outlier.astype(bool)
    else:
        df_res['outlier_preco'] = False
    return df_res

def normalizar_numericos(df: pd.DataFrame) -> pd.DataFrame:
    '''Normaliza colunas numéricas (inteiros e floats) para float64.'''
    df_res = df.copy()
    for col in df_res.columns:
        if pd.api.types.is_numeric_dtype(df_res[col]) and not pd.api.types.is_bool_dtype(df_res[col]):
            df_res[col] = df_res[col].astype('float64')
    return df_res""")

    # Cell 5: Specific Rules
    add_md("""## 4. Regras Específicas por Categoria
- **`memory`**: separa `speed` em `ddr_version` e `speed_mhz` (NaN nos 4 registros com int); separa `modules` em `n_modules`, `gb_per_module` e calcula `total_gb`; valida `ppg_confere`.
- **`internal-hard-drive`**: separa `type` em `drive_type` (SSD, HDD, Desconhecido) e `rpm`; normaliza `form_factor` para `form_factor_norm`.
- **`power-supply`**: padroniza `modular` (False $\\to$ 'Nao modular'); preserva `efficiency` nulo.
- **`cpu-cooler` e `case-fan`**: separa faixas de `rpm`, `noise_level` e `airflow` em `_min` e `_max`.
- **`monitor`**: separa `resolution` em `res_width`, `res_height` e calcula `megapixels`.
- **`cpu`**: cria `gpu_integrada` (booleana baseada em `graphics`).
- **`headphones` e `speakers`**: extrai `freq_resp_min` e `freq_resp_max`.
- **`webcam`**: formata listas de `resolutions` e `os` em strings legíveis.""")
    add_code("""def _extrair_min_max(valor: object) -> tuple[float, float]:
    '''Extrai valores mínimo e máximo a partir de número único ou lista/tupla [min, max].'''
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
    df_res = df.copy()
    ddr_versoes, velocidades_mhz = [], []
    for val in df_res['speed']:
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            ddr_versoes.append(float(val[0]))
            velocidades_mhz.append(float(val[1]))
        elif val is not None and not pd.isna(val):
            ddr_versoes.append(np.nan)
            velocidades_mhz.append(float(val))
        else:
            ddr_versoes.append(np.nan)
            velocidades_mhz.append(np.nan)
    df_res['ddr_version'] = ddr_versoes
    df_res['speed_mhz'] = velocidades_mhz

    n_modulos, gb_por_modulo = [], []
    for val in df_res['modules']:
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            n_modulos.append(float(val[0]))
            gb_por_modulo.append(float(val[1]))
        else:
            n_modulos.append(np.nan)
            gb_por_modulo.append(np.nan)
    df_res['n_modules'] = n_modulos
    df_res['gb_per_module'] = gb_por_modulo
    df_res['total_gb'] = df_res['n_modules'] * df_res['gb_per_module']
    df_res = df_res.drop(columns=['speed', 'modules'])

    tem_dados = df_res['price'].notna() & df_res['price_per_gb'].notna() & df_res['total_gb'].notna() & (df_res['total_gb'] > 0)
    calculado = df_res['price'] / df_res['total_gb']
    df_res['ppg_confere'] = (tem_dados & ((df_res['price_per_gb'] - calculado).abs() <= 0.01)).astype(bool)
    return df_res

def tratar_internal_hard_drive(df: pd.DataFrame) -> pd.DataFrame:
    df_res = df.copy()
    drive_tipos, rpms = [], []
    for t in df_res['type']:
        if pd.isna(t) or t is None:
            drive_tipos.append('Desconhecido')
            rpms.append(np.nan)
        elif str(t).strip().upper() == 'SSD':
            drive_tipos.append('SSD')
            rpms.append(np.nan)
        else:
            drive_tipos.append('HDD')
            try:
                rpms.append(float(t))
            except (ValueError, TypeError):
                rpms.append(np.nan)
    df_res['drive_type'] = drive_tipos
    df_res['rpm'] = rpms

    form_factors_norm = []
    for ff in df_res['form_factor']:
        if pd.isna(ff) or ff is None:
            form_factors_norm.append(None)
        elif isinstance(ff, (int, float)):
            form_factors_norm.append(f'{ff:g}in')
        elif str(ff).strip() in ['2.5', '3.5']:
            form_factors_norm.append(f'{str(ff).strip()}in')
        else:
            form_factors_norm.append(str(ff).strip())
    df_res['form_factor_norm'] = form_factors_norm
    df_res = df_res.drop(columns=['type', 'form_factor'])
    return df_res

def tratar_power_supply(df: pd.DataFrame) -> pd.DataFrame:
    df_res = df.copy()
    modulares = []
    for m in df_res['modular']:
        if m is False or str(m).strip().lower() == 'false':
            modulares.append('Nao modular')
        elif pd.isna(m) or m is None:
            modulares.append(None)
        else:
            modulares.append(str(m).strip())
    df_res['modular'] = modulares
    return df_res

def tratar_cpu_cooler(df: pd.DataFrame) -> pd.DataFrame:
    df_res = df.copy()
    rpm_mins, rpm_maxs = zip(*[_extrair_min_max(v) for v in df_res['rpm']])
    noise_mins, noise_maxs = zip(*[_extrair_min_max(v) for v in df_res['noise_level']])
    df_res['rpm_min'], df_res['rpm_max'] = list(rpm_mins), list(rpm_maxs)
    df_res['noise_level_min'], df_res['noise_level_max'] = list(noise_mins), list(noise_maxs)
    return df_res.drop(columns=['rpm', 'noise_level'])

def tratar_case_fan(df: pd.DataFrame) -> pd.DataFrame:
    df_res = df.copy()
    rpm_mins, rpm_maxs = zip(*[_extrair_min_max(v) for v in df_res['rpm']])
    airflow_mins, airflow_maxs = zip(*[_extrair_min_max(v) for v in df_res['airflow']])
    noise_mins, noise_maxs = zip(*[_extrair_min_max(v) for v in df_res['noise_level']])
    df_res['rpm_min'], df_res['rpm_max'] = list(rpm_mins), list(rpm_maxs)
    df_res['airflow_min'], df_res['airflow_max'] = list(airflow_mins), list(airflow_maxs)
    df_res['noise_level_min'], df_res['noise_level_max'] = list(noise_mins), list(noise_maxs)
    return df_res.drop(columns=['rpm', 'airflow', 'noise_level'])

def tratar_monitor(df: pd.DataFrame) -> pd.DataFrame:
    df_res = df.copy()
    larguras, alturas = [], []
    for val in df_res['resolution']:
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            larguras.append(float(val[0]))
            alturas.append(float(val[1]))
        else:
            larguras.append(np.nan)
            alturas.append(np.nan)
    df_res['res_width'] = larguras
    df_res['res_height'] = alturas
    df_res['megapixels'] = (df_res['res_width'] * df_res['res_height']) / 1_000_000.0
    return df_res.drop(columns=['resolution'])

def tratar_cpu(df: pd.DataFrame) -> pd.DataFrame:
    df_res = df.copy()
    df_res['gpu_integrada'] = df_res['graphics'].notna().astype(bool)
    return df_res

def tratar_headphones(df: pd.DataFrame) -> pd.DataFrame:
    df_res = df.copy()
    freq_mins, freq_maxs = zip(*[_extrair_min_max(v) for v in df_res['frequency_response']])
    df_res['freq_resp_min'], df_res['freq_resp_max'] = list(freq_mins), list(freq_maxs)
    return df_res.drop(columns=['frequency_response'])

def tratar_speakers(df: pd.DataFrame) -> pd.DataFrame:
    df_res = df.copy()
    freq_mins, freq_maxs = zip(*[_extrair_min_max(v) for v in df_res['frequency_response']])
    df_res['freq_resp_min'], df_res['freq_resp_max'] = list(freq_mins), list(freq_maxs)
    return df_res.drop(columns=['frequency_response'])

def tratar_webcam(df: pd.DataFrame) -> pd.DataFrame:
    df_res = df.copy()
    def formatar_lista(val: object) -> str | None:
        if isinstance(val, (list, tuple)):
            return ', '.join(str(item) for item in val if item is not None)
        if val is not None and not pd.isna(val):
            return str(val)
        return None
    df_res['resolutions'] = df_res['resolutions'].apply(formatar_lista)
    df_res['os'] = df_res['os'].apply(formatar_lista)
    return df_res

def tratar_outras_com_listas(df: pd.DataFrame, categoria: str) -> pd.DataFrame:
    df_res = df.copy()
    if categoria == 'os' and 'mode' in df_res.columns:
        df_res['mode'] = df_res['mode'].apply(
            lambda x: '/'.join(str(i) for i in x) if isinstance(x, (list, tuple)) else (str(x) if pd.notna(x) else None)
        )
    elif categoria == 'fan-controller' and 'form_factor' in df_res.columns:
        df_res['form_factor'] = df_res['form_factor'].apply(
            lambda x: f'{x[0]}x {x[1]}in' if isinstance(x, (list, tuple)) else (f'{x:g}in' if isinstance(x, (int, float)) else str(x) if pd.notna(x) else None)
        )
    return df_res

MAPEAMENTO_CATEGORIAS = {
    'memory': tratar_memory,
    'internal-hard-drive': tratar_internal_hard_drive,
    'power-supply': tratar_power_supply,
    'cpu-cooler': tratar_cpu_cooler,
    'case-fan': tratar_case_fan,
    'monitor': tratar_monitor,
    'cpu': tratar_cpu,
    'headphones': tratar_headphones,
    'speakers': tratar_speakers,
    'webcam': tratar_webcam,
}

def aplicar_regras_categoria(df: pd.DataFrame, categoria: str) -> pd.DataFrame:
    if categoria in MAPEAMENTO_CATEGORIAS:
        df = MAPEAMENTO_CATEGORIAS[categoria](df)
    return tratar_outras_com_listas(df, categoria)""")

    # Cell 6: Pipeline Execution
    add_md("""## 5. Execução do Pipeline Completo em Todas as Categorias""")
    add_code("""def processar_categoria(caminho_arquivo: Path, diretorio_saida: Path) -> tuple[dict, pd.DataFrame]:
    categoria = caminho_arquivo.stem
    
    # 1. Leitura
    df_raw = carregar_categoria(caminho_arquivo)
    linhas_lidas = len(df_raw)
    
    # 2. Remoção de duplicatas
    df_limpo, duplicatas = remover_duplicatas(df_raw)
    
    # 3. ID único determinístico
    df_id = gerar_id(df_limpo)
    
    # 4. Extração de marca
    df_marca = extrair_marca(df_id)
    
    # Regras específicas da categoria
    df_cat = aplicar_regras_categoria(df_marca, categoria)
    
    # 5. Preço ausente
    df_preco = adicionar_tem_preco(df_cat)
    
    # 6. Outliers de preço
    df_outlier = calcular_outlier_preco(df_preco)
    
    # 7. Normalização numérica
    df_final = normalizar_numericos(df_outlier)
    
    linhas_finais = len(df_final)
    linhas_com_preco = int(df_final['tem_preco'].sum())
    n_outliers = int(df_final['outlier_preco'].sum())
    pct_com_preco = round((linhas_com_preco / linhas_finais) * 100.0, 2) if linhas_finais > 0 else 0.0
    
    # Exportação Parquet e CSV
    diretorio_saida.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(diretorio_saida / f'{categoria}.parquet', index=False)
    df_final.to_csv(diretorio_saida / f'{categoria}.csv', index=False, encoding='utf-8')
    df_com_preco = df_final[df_final['tem_preco']].copy()
    df_com_preco.to_parquet(diretorio_saida / f'{categoria}_com_preco.parquet', index=False)
    
    metricas = {
        'categoria': categoria,
        'linhas_lidas': linhas_lidas,
        'duplicatas_removidas': duplicatas,
        'linhas_finais': linhas_finais,
        'pct_com_preco': pct_com_preco,
        'outliers_preco': n_outliers,
    }
    return metricas, df_final

arquivos_json = sorted(DIR_INPUT.glob('*.json'))
print(f'Iniciando processamento de {len(arquivos_json)} categorias...\\n')

estatisticas = []
dfs_processados = []

for arq in arquivos_json:
    cat = arq.stem
    metricas, df_proc = processar_categoria(arq, DIR_OUTPUT)
    estatisticas.append(metricas)
    dfs_processados.append(df_proc)
    print(f'[{cat:<22}] Lidas: {metricas["linhas_lidas"]:>5} | Dedup: -{metricas["duplicatas_removidas"]:>3} | Finais: {metricas["linhas_finais"]:>5} | % Preço: {metricas["pct_com_preco"]:>5.1f}% | Outliers: {metricas["outliers_preco"]:>3}')

# Salvar resumo consolidado
df_resumo = pd.DataFrame(estatisticas)[[
    'categoria', 'linhas_lidas', 'duplicatas_removidas', 'linhas_finais', 'pct_com_preco', 'outliers_preco'
]]
caminho_resumo = DIR_OUTPUT / 'resumo_limpeza.csv'
df_resumo.to_csv(caminho_resumo, index=False, encoding='utf-8')
print(f'\\nResumo consolidado salvo em: {caminho_resumo.resolve()}')""")

    # Cell 7: Summary Table
    add_md("""## 6. Tabela de Resumo Consolidada (`resumo_limpeza.csv`)""")
    add_code("""total_lidas = int(df_resumo['linhas_lidas'].sum())
total_dedup = int(df_resumo['duplicatas_removidas'].sum())
total_finais = int(df_resumo['linhas_finais'].sum())
total_outliers = int(df_resumo['outliers_preco'].sum())

print('=== TOTAL GERAL ACUMULADO ===')
print(f'Total de Linhas Lidas:         {total_lidas:,}'.replace(',', '.'))
print(f'Total de Duplicatas Removidas: {total_dedup:,}'.replace(',', '.'))
print(f'Total de Registros Finais:     {total_finais:,}'.replace(',', '.'))
print(f'Total de Outliers de Preço:    {total_outliers:,}'.replace(',', '.'))

display(df_resumo)""")

    # Cell 8: Rare Brands
    add_md("""## 7. Auditoria de Marcas Raras (< 3 ocorrências)
Listagem das marcas extraídas com contagem de frequência inferior a 3 em todo o dataset consolidado:""")
    add_code("""contador_marcas = Counter()
for df_p in dfs_processados:
    if 'marca' in df_p.columns:
        contador_marcas.update(df_p['marca'].dropna().astype(str).tolist())

marcas_raras = [(m, c) for m, c in contador_marcas.items() if c < 3]
marcas_raras.sort(key=lambda x: (x[1], x[0]))

print(f'Total de marcas com menos de 3 ocorrências: {len(marcas_raras)}\\n')
df_raras = pd.DataFrame(marcas_raras, columns=['Marca', 'Ocorrências'])
display(df_raras)""")

    # Cell 9: Quality Asserts
    add_md("""## 8. Verificação e Asserções de Qualidade dos Dados
Bateria completa de validações que garante conformidade com todos os critérios de aceite:""")
    add_code("""print('Executando asserções de validação...')

# 1. Totais globais
assert total_lidas == 66778, f'Esperado 66.778, obtido {total_lidas}'
assert total_dedup == 2951, f'Esperado 2.951, obtido {total_dedup}'
assert total_finais == 63827, f'Esperado 63.827, obtido {total_finais}'

# 2. Unicidade de IDs e Preços não imputados
dict_dfs = {df['categoria'].iloc[0]: df for df in dfs_processados}

for cat_nome, df_p in dict_dfs.items():
    # ID único
    assert df_p['id'].nunique() == len(df_p), f'IDs não são únicos em {cat_nome}!'
    
    # Preços não imputados
    assert df_p['tem_preco'].sum() == df_p['price'].notna().sum()

# 3. Memory: sem listas, 4 registros com int, ppg_confere
df_mem = dict_dfs['memory']
assert df_mem['ddr_version'].isna().sum() == 4, 'Esperado exatamente 4 NaNs em ddr_version!'
linhas_speed_int = df_mem[df_mem['ddr_version'].isna()]
assert set(linhas_speed_int['speed_mhz'].tolist()) == {333.0, 400.0}
assert df_mem['ppg_confere'].dtype == bool
assert df_mem['ppg_confere'].sum() > 0

# 4. Ausência de listas nas categorias especificadas
for cat_check in ['memory', 'monitor', 'cpu-cooler', 'case-fan']:
    df_chk = dict_dfs[cat_check]
    for col in df_chk.columns:
        tem_list = df_chk[col].apply(lambda x: isinstance(x, (list, tuple))).any()
        assert not tem_list, f'Coluna {col} ainda contém listas em {cat_check}!'

# 5. Internal Hard Drive: homogeneidade de tipos e regras
df_ihd = dict_dfs['internal-hard-drive']
assert (df_ihd['drive_type'] == 'Desconhecido').sum() == 30
assert set(df_ihd['drive_type'].unique()) == {'SSD', 'HDD', 'Desconhecido'}
assert df_ihd.loc[df_ihd['drive_type'] == 'HDD', 'rpm'].notna().all()
assert df_ihd.loc[df_ihd['drive_type'] == 'SSD', 'rpm'].isna().all()
for col in df_ihd.columns:
    vals = df_ihd[col].dropna()
    tipos = set(type(x) for x in vals)
    tem_num = any(issubclass(t, (int, float)) for t in tipos)
    tem_str = any(issubclass(t, str) for t in tipos)
    assert not (tem_num and tem_str), f'Coluna {col} em internal-hard-drive mistura texto e número!'

# 6. CPU: gpu_integrada
df_cpu = dict_dfs['cpu']
assert df_cpu['gpu_integrada'].dtype == bool
assert (df_cpu.loc[df_cpu['graphics'].notna(), 'gpu_integrada'] == True).all()

# 7. Power Supply: modular padronizado
df_psu = dict_dfs['power-supply']
assert False not in df_psu['modular'].values
assert 'Nao modular' in df_psu['modular'].values

print('✅ TODAS AS ASSERÇÕES PASSARAM COM SUCESSO (100% VALIDADO)!')""")

    # Salvar em ambos os caminhos (raiz do workspace e dentro de pc_parts_cleaning)
    caminhos = [
        Path("pc_parts_cleaning/pc_part_dataset_cleaning.ipynb"),
        Path("pc_part_dataset_cleaning.ipynb"),
    ]
    for p in caminhos:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
        print(f"Notebook gerado com sucesso em: {p.resolve()}")


if __name__ == "__main__":
    criar_notebook()

