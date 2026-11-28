# PC Part Dataset Cleaning & Preprocessing

Projeto Python para limpeza, padronização, auditoria e pré-processamento do **PC Part Dataset** (extraído do PCPartPicker, com preços em USD).

---

## 1. Visão Geral

O dataset original é composto por 25 arquivos JSON (um por categoria de hardware), totalizando **66.778 registros brutos**. O pipeline realiza:
- Leitura e preservação integral dos arquivos originais em `data/raw/` (somente leitura).
- Deduplicação exata (removendo **2.951 duplicatas**, resultando em **63.827 registros finais**).
- Geração de identificador único (`id`) determinístico via hash SHA-1 de cada registro.
- Extração padronizada de `marca`, com suporte a marcas compostas.
- Sinalização sem imputação de preços ausentes (`tem_preco`).
- Detecção e marcação de outliers de preço por categoria via IQR (`outlier_preco`).
- Tratamento específico para categorias com estruturas complexas (`memory`, `internal-hard-drive`, `cpu-cooler`, `case-fan`, `monitor`, `cpu`, `headphones`, `speakers`, `webcam`).
- Exportação nos formatos **Parquet** e **CSV** (tabela completa e tabela filtrada com preços), além do relatório consolidado `resumo_limpeza.csv`.

---

## 2. Onde fica no projeto

```text
data/raw/              # 25 arquivos JSON originais (somente leitura)
data/processed/        # Saídas geradas (.parquet, .csv, resumo_limpeza.csv)
src/limpeza/
├── load.py            # Leitura dos JSONs brutos
├── common.py          # Etapas comuns (dedup, hash id, marca, outliers, etc.)
├── categories.py      # Transformações específicas de cada categoria
├── report.py          # Relatório consolidado, métricas e auditoria
└── main.py            # Ponto de entrada CLI (argparse)
tests/                 # Testes pytest (totais, IDs, marcas, outliers, categorias)
notebooks/01_limpeza_dataset.ipynb   # Notebook consolidado da limpeza
```

---

## 3. Ambiente

Use o ambiente único da raiz do projeto (ver `README.md` na raiz): `requirements.txt` + `.venv` (Python 3.12).

---

## 4. Como Executar o Pipeline (a partir da raiz)

```bash
# Execução padrão (data/raw -> data/processed)
.venv\Scripts\python -m src.limpeza.main

# Execução com parâmetros customizados
.venv\Scripts\python -m src.limpeza.main --input data/raw --output data/processed
```

Durante a execução, um log no terminal informa o progresso de cada categoria, os totais consolidados e a lista de marcas raras (< 3 ocorrências no dataset) para revisão.

---

## 5. Como Executar os Testes

Para rodar todos os testes automatizados com o `pytest`:

```bash
.venv\Scripts\python -m pytest -v
```

Para rodar testes de uma categoria ou módulo específico:

```bash
python -m pytest tests/test_memory.py -v
python -m pytest tests/test_integration.py -v
```

---

## 6. Etapas de Limpeza e Pré-processamento

### Etapas Comuns (todas as categorias)

1. **Leitura e identificação**: Cada JSON é lido em um DataFrame e recebe a coluna `categoria` (nome do arquivo sem extensão).
2. **Remoção de duplicatas exatas**: Linhas idênticas em todos os campos (incluindo `price`) são eliminadas. Listas e tuplas são serializadas antes da comparação para garantir que estruturas aninhadas sejam comparadas por valor. O volume removido por categoria é contabilizado.
3. **Criação de `id` único**: Gerado como hash SHA-1 hexadecimal de 40 caracteres a partir da serialização determinística (`json.dumps(..., sort_keys=True)`) dos campos da linha deduplicada.
4. **Extração de `marca`**: Extraída a primeira palavra da coluna `name`, respeitando um dicionário canônico de exceções para marcas compostas (ex.: `"Western Digital"`, `"Cooler Master"`, `"Thermal Take"`, `"be quiet!"`, `"Lian Li"`, `"G.Skill"`, `"Silicon Power"`, `"Fractal Design"`, etc.). A coluna `name` original é estritamente preservada.
5. **Tratamento de preços ausentes**: Preços `null` **não são imputados**. É criada a coluna booleana `tem_preco` (`True` quando `price` não for nulo). Duas tabelas são exportadas por categoria: a completa e uma somente com linhas onde `tem_preco == True`.
6. **Detecção de outliers de preço**: Para cada categoria, calculam-se o primeiro quartil ($Q_1$), o terceiro quartil ($Q_3$) e o intervalo interquartil ($IQR = Q_3 - Q_1$) exclusivamente sobre as linhas com preço. A coluna booleana `outlier_preco` marca linhas com $price > Q_3 + 1,5 \times IQR$ ou $price < Q_1 - 1,5 \times IQR$. Linhas sem preço recebem `False`. Nenhuma linha é removida.
7. **Normalização numérica**: Colunas numéricas são convertidas para `float64` para garantir tipagem homogênea e compatibilidade total com o formato Parquet.

---

## 7. Regras Específicas por Categoria e Novas Colunas

| Categoria | Coluna Original | Nova(s) Coluna(s) | Tipo | Descrição / Regra |
|---|---|---|---|---|
| *Todas* | - | `id` | `string` | Hash SHA-1 determinístico do registro. |
| *Todas* | - | `categoria` | `string` | Nome da categoria extraído do arquivo. |
| *Todas* | `name` | `marca` | `string` | Marca extraída via primeira palavra ou regra composta. |
| *Todas* | `price` | `tem_preco` | `bool` | `True` se `price` não for nulo; `False` caso contrário. |
| *Todas* | `price` | `outlier_preco` | `bool` | `True` se o preço for outlier segundo a regra $1,5 \times IQR$. |
| `memory` | `speed` | `ddr_version`<br>`speed_mhz` | `float64`<br>`float64` | `[DDR, MHz]` separado em versão DDR e velocidade em MHz. Nos 4 registros legados onde era inteiro único, `ddr_version = NaN` e `speed_mhz` recebe o valor. Coluna original removida. |
| `memory` | `modules` | `n_modules`<br>`gb_per_module`<br>`total_gb` | `float64`<br>`float64`<br>`float64` | `[n, gb]` separado em contagem de módulos e capacidade unitária; `total_gb = n_modules * gb_per_module`. Coluna original removida. |
| `memory` | `price`, `total_gb`, `price_per_gb` | `ppg_confere` | `bool` | `True` se $|price\_per\_gb - (price / total\_gb)| \le 0,01$; `False` caso contrário ou quando faltam dados. |
| `internal-hard-drive` | `type` | `drive_type`<br>`rpm` | `string`<br>`float64` | `"SSD"` para unidades de estado sólido; `"HDD"` para discos magnéticos; `"Desconhecido"` para os 30 registros nulos. `rpm` contém o RPM numérico para HDDs e `NaN` para SSDs e desconhecidos. Coluna original removida. |
| `internal-hard-drive` | `form_factor` | `form_factor_norm` | `string` | Normaliza dimensões numéricas para polegadas (`2.5` $\to$ `"2.5in"`, `3.5` $\to$ `"3.5in"`) e preserva formatos M.2, mSATA e PCIe em texto. Coluna original removida. |
| `power-supply` | `modular` | `modular` | `string` | Converte o booleano `False` para a string `"Nao modular"`, preservando strings (`"Full"`, `"Semi"`, etc.). `efficiency` nulo é preservado nulo. |
| `cpu-cooler` | `rpm`<br>`noise_level` | `rpm_min`, `rpm_max`<br>`noise_level_min`, `noise_level_max` | `float64`<br>`float64` | Se lista $[min, max]$, separa os extremos. Se número único, $min = max = valor$. Colunas originais de lista removidas. |
| `case-fan` | `rpm`<br>`airflow`<br>`noise_level` | `rpm_min`, `rpm_max`<br>`airflow_min`, `airflow_max`<br>`noise_level_min`, `noise_level_max` | `float64`<br>`float64`<br>`float64` | Separa faixas em mínimo e máximo; número único replica em ambos. Colunas originais de lista removidas. |
| `monitor` | `resolution` | `res_width`<br>`res_height`<br>`megapixels` | `float64`<br>`float64`<br>`float64` | Separa largura e altura da lista $[W, H]$; calcula $megapixels = (res\_width \times res\_height) / 1.000.000$. Coluna original removida. |
| `cpu` | `graphics` | `gpu_integrada` | `bool` | `True` se `graphics` não for nulo; `False` caso contrário. |
| `headphones` | `frequency_response` | `freq_resp_min`<br>`freq_resp_max` | `float64`<br>`float64` | Separa a faixa de resposta de frequência $[min, max]$ em colunas numéricas escalares. Coluna original removida. |
| `speakers` | `frequency_response` | `freq_resp_min`<br>`freq_resp_max` | `float64`<br>`float64` | Separa a faixa de resposta de frequência $[min, max]$ em colunas numéricas escalares. Coluna original removida. |
| `webcam` | `resolutions`<br>`os` | `resolutions`<br>`os` | `string`<br>`string` | Converte listas de strings em texto legível separado por vírgula (ex.: `"4k, 1080p, 720p"`, `"Windows, OS X"`). |

---

## 8. Decisões de Design e Ambiguidade Documentadas

### 8.1. `frequency_response` em Headphones e Speakers
- **Ambiguidade identificada**: A documentação original (`API.md`) cita a unidade de `frequency_response` como `kHz`. No entanto, os dados brutos apresentam valores como `[15, 25]`, `[20, 20]`, `[100, 17]`, `[35, 20]`, onde em certos casos o primeiro elemento é em Hz e o segundo em kHz, ou faixas nominais audíveis abreviadas. Além disso, a ordem dos elementos nos alto-falantes por vezes inverte extremos.
- **Decisão adotada**: Para não introduzir distorções arbitrárias nos dados brutos através de fatores de conversão incertos (multiplicações por 1.000), os valores numéricos foram extraídos diretamente como `freq_resp_min` e `freq_resp_max` correspondentes à posição $[0]$ e $[1]$ da lista original. As colunas originais aninhadas foram removidas para garantir suporte nativo a esquemas relacionais e Parquet.

### 8.2. Colunas de Lista em Webcam (`resolutions`, `os`)
- Em vez de gerar dezenas de colunas esparsas booleanas para cada sistema operacional e resolução, as listas foram padronizadas como strings legíveis separadas por vírgula (`"Windows, OS X, ChromeOS"`). Isso preserva a informação completa de compatibilidade sem inflar desnecessariamente o schema e mantendo compatibilidade direta com Parquet e CSV.

### 8.3. Normalização de `form_factor` em `internal-hard-drive`
- Na fonte original, discos mecânicos vinham com valores float `2.5` e `3.5`, enquanto SSDs modernos vinham com strings como `"M.2-2280"` ou `"PCIe"`. A padronização para strings com o sufixo `"in"` (`"2.5in"`, `"3.5in"`) assegura que a coluna `form_factor_norm` possua um único tipo consistente (texto), eliminando tipos mistos.

### 8.4. Tratamento de Outliers
- Outliers de preço são sinalizados via coluna booleana `outlier_preco`, mas **não são excluídos da base**. Essa abordagem preserva hardware topo de linha ou edições limitadas legítimas (como memórias de alta capacidade ou GPUs profissionais), permitindo que análises posteriores filtrem ou mantenham tais registros conforme o objetivo analítico.

---

## 9. Saídas Geradas

Ao término da execução em `data/processed/`, são gerados:
1. `<categoria>.parquet` e `<categoria>.csv`: Tabela final tratada completa (todos os registros deduplicados da categoria).
2. `<categoria>_com_preco.parquet`: Subconjunto contendo apenas produtos com preço informado (`tem_preco == True`).
3. `resumo_limpeza.csv`: Tabela resumo com métricas consolidadas por categoria:
   - `categoria`: Nome da categoria
   - `linhas_lidas`: Quantidade de registros originais no arquivo JSON
   - `duplicatas_removidas`: Quantidade de duplicatas exatas eliminadas
   - `linhas_finais`: Registros únicos resultantes
   - `pct_com_preco`: Percentual de linhas com preço informado
   - `outliers_preco`: Quantidade de outliers detectados via critério $1,5 \times IQR$

