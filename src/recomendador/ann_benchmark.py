"""Busca exata x busca aproximada (ANN): ANN Recall@10, latência, memória.

1) Catálogo completo real (25 categorias, ~63 mil itens).
2) Teste de escala: catálogo sintético de 500 mil vetores (itens reais + ruído),
   para mostrar quando o ANN passa a compensar.

Uso:  python -m src.recomendador.ann_benchmark
"""
from __future__ import annotations

import time

import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import embeddings as E
from .consultas import gerar_consultas
from .dados import RAIZ, carregar_catalogo, carregar_catalogo_completo
from .textos import texto_consulta, texto_generico

K = 10
RES = RAIZ / "resultados"
faiss.omp_set_num_threads(1)  # latência por consulta, comparação justa


def medir(idx, xq, verdade, nome, param, t_build):
    lat, achados = [], []
    for q in xq:
        t = time.perf_counter()
        _, I = idx.search(q[None, :], K)
        lat.append((time.perf_counter() - t) * 1000)
        achados.append(I[0])
    rec = np.mean([len(set(a) & set(v)) / K for a, v in zip(achados, verdade)])
    return {"indice": nome, "parametro": param, "ANN Recall@10": rec,
            "latencia_media_ms": np.mean(lat), "latencia_p95_ms": np.percentile(lat, 95),
            "construcao_s": t_build, "tamanho_MB": E.tamanho_mb(idx)}


def varredura(x, xq, rotulo):
    linhas = []
    t = time.perf_counter(); flat = E.indice_exato(x); tb = time.perf_counter() - t
    _, verdade = flat.search(xq, K)
    linhas.append(medir(flat, xq, verdade, "Exata (Flat)", "-", tb))

    t = time.perf_counter(); h = E.indice_hnsw(x, M=32, ef_construcao=200); tb = time.perf_counter() - t
    for ef in [8, 16, 32, 64, 128, 256]:
        h.hnsw.efSearch = ef
        linhas.append(medir(h, xq, verdade, "HNSW (M=32)", f"efSearch={ef}", tb))

    t = time.perf_counter(); ivf = E.indice_ivf(x); tb = time.perf_counter() - t
    for npb in [1, 2, 4, 8, 16, 32, 64]:
        ivf.nprobe = npb
        linhas.append(medir(ivf, xq, verdade, f"IVF (nlist={ivf.nlist})", f"nprobe={npb}", tb))

    t = time.perf_counter(); pq = E.indice_ivfpq(x); tb = time.perf_counter() - t
    for npb in [4, 16, 64]:
        pq.nprobe = npb
        linhas.append(medir(pq, xq, verdade, "IVF-PQ (48x8 bits)", f"nprobe={npb}", tb))
    df = pd.DataFrame(linhas)
    df.insert(0, "catalogo", rotulo)
    return df


def grafico(df, arq, titulo):
    fig, ax = plt.subplots(figsize=(8, 5))
    cores = {"HNSW": "#c2571a", "IVF (": "#2f6db3", "IVF-PQ": "#5b9c4a"}
    for nome, g in df[df.indice != "Exata (Flat)"].groupby("indice"):
        cor = next(v for k, v in cores.items() if nome.startswith(k))
        ax.plot(g["latencia_media_ms"], g["ANN Recall@10"], "o-", color=cor, label=nome)
    exata = df[df.indice == "Exata (Flat)"].iloc[0]
    ax.axvline(exata["latencia_media_ms"], color="#555", ls="--",
               label=f"Exata: {exata['latencia_media_ms']:.2f} ms")
    ax.set_xscale("log")
    ax.set_xlabel("Latência média por consulta (ms, escala log, 1 thread)")
    ax.set_ylabel("ANN Recall@10 (vs. busca exata)")
    ax.set_title(titulo)
    ax.legend(); ax.grid(alpha=.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(arq, dpi=150); plt.close(fig)


def main():
    RES.mkdir(exist_ok=True)
    nome_m = E.nome_modelo("v2" if E.caminho_modelo("v2").exists() else "base")
    print("Modelo:", nome_m)
    todo = carregar_catalogo_completo()
    x = E.embed([texto_generico(r) for _, r in todo.iterrows()], "catalogo_completo",
                modelo_nome=nome_m)
    qs = gerar_consultas(carregar_catalogo())
    xq = E.embed([texto_consulta(q["alvo"], q["montagem"]) for q in qs], "consultas_ann",
                 modelo_nome=nome_m)
    print(f"Catálogo: {len(x)} vetores de {x.shape[1]} dimensões | {len(xq)} consultas")

    real = varredura(x, xq, f"real ({len(x)//1000} mil)")
    print(real.round(4).to_string(index=False))
    grafico(real, RES / "ann_tradeoff_real.png", f"Exata x ANN — catálogo real ({len(x):,} itens)")

    # Escala: 500 mil vetores sintéticos (itens reais perturbados)
    rng = np.random.default_rng(0)
    n = 500_000
    base = x[rng.integers(0, len(x), n)]
    xs = base + rng.normal(0, 0.03, base.shape).astype("float32")
    xs /= np.linalg.norm(xs, axis=1, keepdims=True)
    esc = varredura(xs.astype("float32"), xq[:300], "sintético (500 mil)")
    print(esc.round(4).to_string(index=False))
    grafico(esc, RES / "ann_tradeoff_500mil.png", "Exata x ANN — catálogo sintético de 500 mil itens")

    pd.concat([real, esc]).round(4).to_csv(RES / "ann_benchmark.csv", index=False)
    print("Resultados em", RES)


if __name__ == "__main__":
    main()
