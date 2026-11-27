"""Avaliação offline: recuperação, ranking e qualidade da lista.

Uso:  python -m src.avaliacao
Saídas em resultados/: tabelas CSV/Markdown e gráficos.
"""
from __future__ import annotations

import itertools
import json
import time
import zlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import embeddings as E
from .baselines import Lexical, ordenar_por_score, popularidade
from .compat import mascara_compativel, percentil_cpu_na_plataforma, relevancia
from .dados import RAIZ
from .recomendador import Pesos, Recomendador, ancora, features, ordenar
from .textos import texto_consulta, textos

RES = RAIZ / "resultados"
ALVOS = ["motherboard", "memory", "case", "power-supply"]
# Divisão "temporal" por geração de plataforma (o dataset não tem datas)
PLAT_TREINO = ["LGA1151", "AM4", "LGA1200"]          # lançadas até 2020
PLAT_TESTE = ["LGA1700", "AM5", "LGA1851"]           # lançadas de 2021 em diante
K = 10
K_CAND = 100


# ---------------------------------------------------------------------------
# Consultas de avaliação
# ---------------------------------------------------------------------------
def gerar_consultas(cat: dict[str, pd.DataFrame], seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    cpus = cat["cpu"]
    cpus = cpus[cpus["socket"].isin(PLAT_TREINO + PLAT_TESTE)].drop_duplicates("name").copy()
    cpus["pct_plat"] = percentil_cpu_na_plataforma(cpus)
    gpus = cat["video-card"]
    qs = []
    for _, cpu in cpus.iterrows():
        split = "treino" if cpu["socket"] in PLAT_TREINO else "teste"
        # treino é dividido por CPU: 70% para o fine-tuning, 30% para ajustar os pesos
        if split == "teste":
            grupo = "teste"
        else:
            grupo = "ajuste" if zlib.crc32(cpu["name"].encode()) % 10 < 7 else "validacao"
        pct = float(cpu["pct_plat"])
        base = {"cpu": cpu}
        mbs = cat["motherboard"][mascara_compativel("motherboard", cat["motherboard"], base)]
        if mbs.empty:
            continue
        mb = mbs.iloc[rng.integers(len(mbs))]
        perto = (gpus["preco_pct"] - cpu["preco_pct"]).abs().nsmallest(10).index
        gpu = gpus.loc[rng.choice(perto)]
        for alvo, mont in [("motherboard", {"cpu": cpu}),
                           ("memory", {"cpu": cpu, "motherboard": mb}),
                           ("case", {"cpu": cpu, "motherboard": mb}),
                           ("power-supply", {"cpu": cpu, "video-card": gpu})]:
            g = relevancia(alvo, cat[alvo], mont, pct)
            if (g > 0).sum() == 0:
                continue
            qs.append({"alvo": alvo, "montagem": mont, "pct": pct, "split": grupo,
                       "cpu": cpu["name"], "socket": cpu["socket"], "grades": g})
    return qs


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def dcg(g: np.ndarray) -> float:
    g = np.asarray(g, dtype=float)
    return float(((2 ** g - 1) / np.log2(np.arange(2, len(g) + 2))).sum())


def ndcg(lista: np.ndarray, grades: np.ndarray, k: int = K) -> float:
    ideal = dcg(np.sort(grades)[::-1][:k])
    return dcg(grades[lista[:k]]) / ideal if ideal > 0 else 0.0


def mrr(lista: np.ndarray, grades: np.ndarray, k: int = K) -> float:
    for i, j in enumerate(lista[:k]):
        if grades[j] == 2:
            return 1.0 / (i + 1)
    return 0.0


def metricas(lista: np.ndarray, grades: np.ndarray, df: pd.DataFrame) -> dict:
    top = lista[:K]
    rel = grades > 0
    return {
        "Compat@10": float(rel[top].mean()) if len(top) else 0.0,
        "Recall@100": float(rel[lista[:K_CAND]].sum() / rel.sum()),
        "nDCG@10": ndcg(lista, grades),
        "MRR": mrr(lista, grades),
        "Diversidade@10": df["marca"].iloc[top].nunique() / K if len(top) else 0.0,
        "Itens na lista": len(top),
    }


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------
def main():
    RES.mkdir(exist_ok=True)
    t0 = time.time()
    rec = Recomendador(variante="enriquecido", tipo_indice="hnsw", k_candidatos=K_CAND,
                       ajustado=True, pesos=Pesos())
    if not rec.ajustado:
        raise SystemExit("Rode antes: python -m src.finetune")
    cat = rec.cat
    bruto = {c: E.embed(textos(c, cat[c], "bruto"), f"{c}_bruto_base") for c in ALVOS}
    base = {c: E.embed(textos(c, cat[c], "enriquecido"), f"{c}_enriquecido_base") for c in ALVOS}
    lex = Lexical({c: textos(c, cat[c], "enriquecido") for c in ALVOS})
    qs = gerar_consultas(cat)
    print(f"{len(qs)} consultas | treino={sum(q['split']=='treino' for q in qs)} "
          f"teste={sum(q['split']=='teste' for q in qs)} | setup {time.time()-t0:.0f}s")

    linhas_a, linhas_b, latencias = [], [], {}
    cache_cand = []

    def reg_lat(nome, dt):
        latencias.setdefault(nome, []).append(dt * 1000)

    for qi, q in enumerate(qs):
        alvo, mont, g = q["alvo"], q["montagem"], q["grades"]
        df = cat[alvo]
        valido = mascara_compativel(alvo, df, mont)
        desemp = df["pop_rank"].to_numpy()
        texto_q = texto_consulta(alvo, mont, "enriquecido")
        meta = {k: q[k] for k in ("alvo", "split", "cpu", "socket")}

        scores = {}
        t = time.perf_counter(); scores["Popularidade"] = popularidade(df); reg_lat("Popularidade", time.perf_counter() - t)
        t = time.perf_counter(); scores["Palavra-chave"] = lex.palavra_chave(alvo, texto_q); reg_lat("Palavra-chave", time.perf_counter() - t)
        t = time.perf_counter(); scores["BM25"] = lex.bm25_scores(alvo, texto_q); reg_lat("BM25", time.perf_counter() - t)
        t = time.perf_counter()
        vq_b = E.embed([texto_consulta(alvo, mont, "bruto")])
        scores["Neural pré-treinado (texto bruto)"] = bruto[alvo] @ vq_b[0]
        reg_lat("Neural pré-treinado (texto bruto)", time.perf_counter() - t)
        t = time.perf_counter()
        vq_e = E.embed([texto_q])
        scores["Neural pré-treinado (texto enriquecido)"] = base[alvo] @ vq_e[0]
        reg_lat("Neural pré-treinado (texto enriquecido)", time.perf_counter() - t)
        t = time.perf_counter()
        vq = rec.vetor_consulta(alvo, mont)
        scores["Neural ajustado (fine-tuning)"] = rec.vetores[alvo] @ vq[0]
        reg_lat("Neural ajustado: busca exata", time.perf_counter() - t)

        for nome, s in scores.items():
            # Tabela A: recuperação pura (sem filtro de compatibilidade)
            lista = ordenar_por_score(s, desemp, K_CAND)
            linhas_a.append({**meta, "metodo": nome, **metricas(lista, g, df)})
            # Tabela B: lista final com filtro de compatibilidade aplicado
            lista_f = ordenar_por_score(s, desemp, K, valido)
            linhas_b.append({**meta, "metodo": nome + " + filtro", **metricas(lista_f, g, df)})

        # Sistema: candidatos via HNSW + ranqueamento
        t = time.perf_counter()
        sims, ids = rec.indices[alvo].search(vq, min(K_CAND, len(df)))
        reg_lat("Neural ajustado: HNSW (só a busca)", time.perf_counter() - t)
        cand, sims = ids[0][ids[0] >= 0], sims[0][ids[0] >= 0]
        f = features(df, cand, sims, ancora(mont))
        cache_cand.append((q, cand, f, valido[cand], meta))

    # ---------- ajuste dos pesos do ranking só na VALIDAÇÃO (CPUs não usadas no fine-tuning) ----------
    grade_busca = list(itertools.product([0, 0.5, 1, 2], [0, 0.5, 1], [0, 1, 2, 4], [0, 0.3, 1]))

    def avalia(p: Pesos, split: str) -> float:
        v = [ndcg(cand[ordenar(f, p, K, val)], q["grades"])
             for q, cand, f, val, _ in cache_cand if q["split"] == split]
        return float(np.mean(v))

    resultados_grade = []
    for ws, wp, wc, mm in grade_busca:
        p = Pesos(sim=ws, pop=wp, coer=wc, outlier=0.5, mmr=mm)
        resultados_grade.append((avalia(p, "validacao"), ws, wp, wc, mm))
    melhor = max(resultados_grade)
    pesos = Pesos(sim=melhor[1], pop=melhor[2], coer=melhor[3], outlier=0.5, mmr=melhor[4])
    sem_sim = max(r for r in resultados_grade if r[1] == 0)
    pesos_sem_sim = Pesos(sim=0, pop=sem_sim[2], coer=sem_sim[3], outlier=0.5, mmr=sem_sim[4])
    sem_coer = max(r for r in resultados_grade if r[3] == 0)
    pesos_sem_coer = Pesos(sim=sem_coer[1], pop=sem_coer[2], coer=0, outlier=0.5, mmr=sem_coer[4])
    print("Pesos escolhidos na validação:", pesos, f"nDCG validação={melhor[0]:.3f}")
    pd.DataFrame(resultados_grade, columns=["nDCG_validacao", "sim", "pop", "coer", "mmr"])         .sort_values("nDCG_validacao", ascending=False).to_csv(RES / "busca_pesos.csv", index=False)
    (RES / "pesos.json").write_text(json.dumps(pesos.__dict__, indent=2))

    for q, cand, f, val, meta in cache_cand:
        df = cat[q["alvo"]]
        for nome, p in [("Sistema: Neural + ranking", pesos),
                        ("Ablação: ranking sem similaridade neural", pesos_sem_sim),
                        ("Ablação: ranking sem coerência de preço", pesos_sem_coer),
                        ("Ablação: Neural + filtro, sem ranking", Pesos(1, 0, 0, 0, 0))]:
            lista = cand[ordenar(f, p, K, val)]
            linhas_b.append({**meta, "metodo": nome, **metricas(lista, q["grades"], df)})

    a = pd.DataFrame(linhas_a)
    b = pd.DataFrame(linhas_b)
    a.to_csv(RES / "por_consulta_recuperacao.csv", index=False)
    b.to_csv(RES / "por_consulta_lista_final.csv", index=False)

    colsA = ["Compat@10", "Recall@100", "nDCG@10"]
    colsB = ["nDCG@10", "MRR", "Compat@10", "Diversidade@10", "Itens na lista"]
    for split in ["validacao", "teste"]:
        ta = a[a.split == split].groupby("metodo")[colsA].mean().sort_values("nDCG@10")
        tb = b[b.split == split].groupby("metodo")[colsB].mean().sort_values("nDCG@10")
        ta.round(3).to_csv(RES / f"tabela_recuperacao_{split}.csv")
        tb.round(3).to_csv(RES / f"tabela_lista_final_{split}.csv")
        print(f"\n=== RECUPERAÇÃO sem filtro ({split}) ===\n{ta.round(3).to_string()}")
        print(f"\n=== LISTA FINAL ({split}) ===\n{tb.round(3).to_string()}")

    por_cat = b[b.split == "teste"].pivot_table(index="metodo", columns="alvo", values="nDCG@10").round(3)
    por_cat.to_csv(RES / "ndcg_por_categoria_teste.csv")
    print(f"\n=== nDCG@10 por categoria (teste) ===\n{por_cat.to_string()}")

    # Intervalo de confiança (bootstrap pareado) sistema vs melhor baseline no teste
    bt = b[b.split == "teste"]
    sis = bt[bt.metodo == "Sistema: Neural + ranking"]["nDCG@10"].to_numpy()
    rivais = bt[bt.metodo.str.endswith("+ filtro")].groupby("metodo")["nDCG@10"].mean()
    melhor_base = rivais.idxmax()
    base = bt[bt.metodo == melhor_base]["nDCG@10"].to_numpy()
    rng = np.random.default_rng(0)
    dif = sis - base
    boots = [dif[rng.integers(0, len(dif), len(dif))].mean() for _ in range(5000)]
    ic = (float(np.mean(dif)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    print(f"\nSistema - {melhor_base}: Δ nDCG@10 = {ic[0]:.3f} (IC95% {ic[1]:.3f} a {ic[2]:.3f})")

    lat = pd.DataFrame({m: [np.mean(v), np.percentile(v, 95)] for m, v in latencias.items()},
                       index=["média (ms)", "p95 (ms)"]).T.round(3)
    lat.to_csv(RES / "latencia_metodos.csv")
    print(f"\n=== Latência da etapa de pontuação/recuperação ===\n{lat.to_string()}")

    resumo = {"n_consultas": len(qs), "pesos": pesos.__dict__,
              "melhor_baseline_teste": melhor_base,
              "delta_ndcg_teste": ic[0], "ic95": [ic[1], ic[2]]}
    (RES / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False))

    # ---------- gráficos ----------
    tb = b[b.split == "teste"].groupby("metodo")["nDCG@10"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    cores = ["#c2571a" if m.startswith("Sistema") else "#8a8f98" for m in tb.index]
    ax.barh(tb.index, tb.values, color=cores)
    for i, v in enumerate(tb.values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("nDCG@10 (teste: AM5, LGA1700, LGA1851)")
    ax.set_title("Qualidade da lista final por método")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(RES / "ndcg_lista_final.png", dpi=150); plt.close(fig)

    ta = a[a.split == "teste"].groupby("metodo")["Compat@10"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(ta.index, ta.values, color=["#c2571a" if "ajustado" in m else "#8a8f98" for m in ta.index])
    for i, v in enumerate(ta.values):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlabel("Fração de itens compatíveis no top-10, SEM filtro (teste)")
    ax.set_title("Geração de candidatos: o método sozinho acerta a compatibilidade?")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(RES / "compat_sem_filtro.png", dpi=150); plt.close(fig)
    print(f"\nConcluído em {time.time()-t0:.0f}s. Resultados em {RES}")


if __name__ == "__main__":
    main()
