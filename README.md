# Recomendador de peças de PC compatíveis (Neural Search)

Projeto final de **Sistemas de Recomendação**. O usuário escolhe uma peça (ex.: um *AMD Ryzen 5 5600*)
e o sistema recomenda, passo a passo, as demais peças **compatíveis** — placa-mãe, memória, GPU,
cooler, fonte, gabinete e armazenamento — usando embeddings, índice vetorial (ANN) e ranqueamento
em duas etapas. Enunciado em [`docs/A2_enunciado.pdf`](docs/A2_enunciado.pdf); plano e progresso em
[`docs/Plano_e_Progresso.pdf`](docs/Plano_e_Progresso.pdf).

## Estrutura

```text
.
├── data/
│   ├── raw/             25 JSON originais do PCPartPicker (somente leitura)
│   └── processed/       catálogo limpo (.parquet/.csv) + resumo_limpeza.csv
├── src/
│   ├── limpeza/         pipeline de limpeza e pré-processamento
│   └── recomendador/    sistema de recomendação
│       ├── dados.py         carga do catálogo + atributos derivados (socket, DDR, formatos, consumo)
│       ├── compat.py        regras de compatibilidade e relevância graduada (0/1/2)
│       ├── textos.py        peças e montagem -> texto
│       ├── embeddings.py    embeddings (com cache) e índices FAISS (Flat, HNSW, IVF, IVF-PQ)
│       ├── recomendador.py  candidatos (ANN) + ranking + montagem sequencial
│       ├── baselines.py     popularidade, palavra-chave, BM25
│       ├── finetune.py      fine-tuning contrastivo do modelo de embeddings
│       ├── avaliacao.py     matriz de avaliação, ablações, bootstrap, gráficos
│       ├── ann_benchmark.py busca exata x ANN (recall, latência, memória)
│       └── demo.py          demonstração no terminal
├── tests/               testes da limpeza (pytest)
├── notebooks/           01_limpeza_dataset.ipynb
├── resultados/          tabelas, gráficos e pesos gerados pela avaliação
├── docs/                enunciado, documentação do dataset e da limpeza, plano (PDF)
├── scripts/             gerar_documentacao.py (gera o PDF do plano)
├── cache/               embeddings em cache (gerado, fora do git)
└── modelos/             modelo ajustado por fine-tuning (gerado, fora do git)
```

## Ambiente

```bash
uv venv --python 3.12 .venv          # ou: python -m venv .venv
uv pip install -r requirements.txt   # ou: .venv\Scripts\pip install -r requirements.txt
```

## Como executar (sempre a partir da raiz)

| Comando | O que faz |
|---|---|
| `.venv\Scripts\python -m src.limpeza.main` | Regera `data/processed` a partir de `data/raw` |
| `.venv\Scripts\python -m pytest -q` | Testes da limpeza |
| `.venv\Scripts\python -m src.recomendador.finetune` | Treina o modelo de embeddings (salva em `modelos/`) |
| `.venv\Scripts\python -m src.recomendador.avaliacao` | Avaliação completa (salva em `resultados/`) |
| `.venv\Scripts\python -m src.recomendador.ann_benchmark` | Busca exata x ANN |
| `.venv\Scripts\python -m src.recomendador.demo "Ryzen 5 5600"` | Monta um PC a partir da CPU |
| `.venv\Scripts\python -m src.recomendador.demo --busca "processador barato para jogos"` | Busca livre em português |
| `.venv\Scripts\python scripts\gerar_documentacao.py` | Regera o PDF do plano |

## Dados

Dataset original: [pc-part-dataset](https://github.com/docyx/pc-part-dataset) (PCPartPicker, preços em USD,
licença MIT — ver `docs/dataset_LICENSE.txt` e campos em `docs/dataset_campos_API.md`).
Detalhes da limpeza em [`docs/limpeza_dados.md`](docs/limpeza_dados.md).
