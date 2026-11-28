"""Avaliação offline em camadas: embeddings, candidatos, ranking e qualidade da lista.

Protocolo (nada do teste é usado para treinar):
  ajuste    (70% das CPUs de plataformas antigas) -> fine-tuning dos embeddings
  validacao (30% das CPUs de plataformas antigas) -> treino do LambdaMART
  teste     (AM5, LGA1700, LGA1851)               -> números reportados

Uso:  python -m src.recomendador.avaliacao
"""
from __future__ import annotations

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import embeddings as E
from . import ranking as R
from .baselines import ordenar_por_score, popularidade
from .compat import mascara_compativel
from .consultas import ALVOS, gerar_consultas
from .dados import RAIZ
from .metricas import K, K_CAND, bootstrap_dif, metricas
from .recomendador import Recomendador
from .recuperacao import hibrido
from .textos import texto_consulta, textos

RES = RAIZ / "resultados"
SISTEMA = "Sistema: híbrido filtrado + LambdaMART + MMR"
RECENTES_TREINO = {"AM4", "LGA1151", "LGA1200", "LGA2066"}
COR_SIS, COR_NEU, COR_BASE = "#c2571a", "#e39a5f", "#8a8f98"


def _cor(m: str) -> str:
    if m.startswith("Sistema"):
        return COR_SIS
    return COR_NEU if ("Neural" in m or "Híbrido" in m) else COR_BASE


def barras(serie: pd.Series, arq, xlabel: str, titulo: str, fmt="{:.3f}"):
    serie = serie.sort_values()
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(serie) + 1.4))
    ax.barh(serie.index, serie.values, color=[_cor(m) for m in serie.index])
    for i, v in enumerate(serie.values):
        ax.text(v + 0.005, i, fmt.format(v), va="center", fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_title(titulo)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(arq, dpi=150); plt.close(fig)


def main():
    RES.mkdir(exist_ok=True)
    t0 = time.time()
    rec = Recomendador(versao_modelo="v2", usar_ranker=False)
    if rec.versao_modelo != "v2":
        raise SystemExit("Rode antes: python -m src.recomendador.finetune --versao v2")
    cat = rec.cat

    # Modelos de embedding comparados: (vetores do catálogo, nome do modelo, variante do texto)
    modelos = {
        "Neural pré-treinado (texto bruto)": ("base", "bruto"),
        "Neural pré-treinado": ("base", "enriquecido"),
        "Neural ajustado v1": ("v1", "enriquecido"),
        "Neural ajustado v2": ("v2", "enriquecido"),
    }
    vets = {}
    for nome, (versao, var) in list(modelos.items()):
        if versao != "base" and not E.caminho_modelo(versao).exists():
            modelos.pop(nome)
            continue
        vets[nome] = {c: E.embed(textos(c, cat[c], var), f"{c}_{var}_{versao}",
                                 modelo_nome=E.nome_modelo(versao)) for c in ALVOS}

    qs = gerar_consultas(cat)
    grupos = {g: sum(q["split"] == g for q in qs) for g in ("ajuste", "validacao", "teste")}
    print(f"{len(qs)} consultas {grupos} | setup {time.time()-t0:.0f}s")

    linhas_a, linhas_b, cache = [], [], []
    for q in qs:
        if q["split"] == "ajuste":
            continue  # já usadas no fine-tuning
        alvo, mont, g = q["alvo"], q["montagem"], q["grades"]
        df = cat[alvo]
        valido = mascara_compativel(alvo, df, mont)
        todos = np.ones(len(df), bool)
        desemp = df["pop_rank"].to_numpy()
        tq = texto_consulta(alvo, mont)
        meta = {k: q[k] for k in ("alvo", "split", "cpu", "socket")}

        scores = {"Popularidade": popularidade(df),
                  "Palavra-chave": rec.lex.palavra_chave(alvo, tq),
                  "BM25": rec.lex.bm25_scores(alvo, tq)}
        for nome, (versao, var) in modelos.items():
            vq = E.embed([texto_consulta(alvo, mont, var)], modelo_nome=E.nome_modelo(versao))
            scores[nome] = vets[nome][alvo] @ vq[0]

        for nome, s in scores.items():
            linhas_a.append({**meta, "metodo": nome,
                             **metricas(ordenar_por_score(s, desemp, K_CAND), g, df)})
            linhas_b.append({**meta, "metodo": nome + " + filtro",
                             **metricas(ordenar_por_score(s, desemp, K, valido), g, df)})

        # Híbrido sem filtro (só recuperação): BM25 + neural v2 por RRF
        cand_h, *_ = hibrido(scores["Neural ajustado v2"], scores["BM25"], todos, K_CAND)
        linhas_a.append({**meta, "metodo": "Híbrido (BM25 + neural v2, RRF)",
                         **metricas(cand_h, g, df)})

        # Sistema: candidatos híbridos filtrados (+ variáveis para o ranker)
        c = rec.candidatos(alvo, mont)
        linhas_b.append({**meta, "metodo": "Híbrido filtrado (RRF), sem ranker",
                         **metricas(c["cand"], g, df)})
        _, X = rec.pontuar(alvo, mont, c)
        cache.append({"q": q, "meta": meta, "cand": c["cand"], "X": X,
                      "y": g[c["cand"]], "marcas": df["marca"].to_numpy()[c["cand"]]})
    print(f"Candidatos e variáveis prontos ({time.time()-t0:.0f}s)")

    # ---------------- LambdaMART: seleção por validação TEMPORAL (sem tocar no teste) -------
    # Dentro da validação: treina nas plataformas mais antigas e valida nas mais recentes
    # delas (AM4, LGA1151, LGA1200, LGA2066), imitando o salto "antigo -> novo" do teste.
    val = [d for d in cache if d["q"]["split"] == "validacao" and len(d["cand"])]
    v_ant = [d for d in val if d["q"]["socket"] not in RECENTES_TREINO]
    v_rec = [d for d in val if d["q"]["socket"] in RECENTES_TREINO]

    def empilha(ds):
        return (pd.concat([d["X"] for d in ds], ignore_index=True),
                np.concatenate([d["y"] for d in ds]), [len(d["cand"]) for d in ds])

    def ndcg_medio(ds, m, lam):
        v = []
        for d in ds:
            s_ = m.predict(d["X"][m.features_]) if m is not None else d["X"]["rrf"].to_numpy()
            v.append(metricas(d["cand"][R.mmr_marcas(s_, d["marcas"], K, lam)],
                              d["q"]["grades"], cat[d["q"]["alvo"]])["nDCG@10"])
        return float(np.mean(v))

    sem = lambda grupo: [f for f in R.FEATURES if f not in R.GRUPOS[grupo]]
    conjuntos = {"todas": R.FEATURES, **{f"sem {g}": sem(g) for g in
                 ("neural", "lexical", "preco", "relacionais", "popularidade")}}
    Xa, ya, ga = empilha(v_ant)
    sel = [{"config": "RRF (sem ranker)", "variaveis": "-", "hiper": "-", "mmr": lam,
            "nDCG@10 val. temporal": ndcg_medio(v_rec, None, lam)} for lam in (0.0, 0.3)]
    for nome_c, feats in conjuntos.items():
        for h in R.HIPER:
            m = R.treinar(Xa, ya, ga, feats, h)
            for lam in (0.0, 0.3):
                sel.append({"config": "LambdaMART", "variaveis": nome_c, "hiper": h, "mmr": lam,
                            "nDCG@10 val. temporal": ndcg_medio(v_rec, m, lam)})
    sel = pd.DataFrame(sel).sort_values("nDCG@10 val. temporal", ascending=False)
    sel.round(4).to_csv(RES / "selecao_ranker.csv", index=False)
    print("\n=== Seleção do ranker (treino: plataformas antigas; validação: "
          "AM4/LGA1151/LGA1200/LGA2066) ===\n" + sel.head(12).round(4).to_string(index=False))
    melhor_lmart = sel[sel.config == "LambdaMART"].iloc[0]
    hiper, vars_ = melhor_lmart["hiper"], conjuntos[melhor_lmart["variaveis"]]
    lam_sis = 0.3  # diversidade é requisito de produto; o custo em nDCG aparece na ablação

    # Treino final do LambdaMART em toda a validação com a configuração escolhida
    Xtr, ytr, gtr = empilha(val)
    principal = R.treinar(Xtr, ytr, gtr, vars_, hiper)
    R.salvar(principal)
    print(f"LambdaMART final: variáveis={melhor_lmart['variaveis']}, hiper={hiper}; "
          f"{len(val)} consultas / {len(Xtr)} pares; salvo em {R.ARQ_RANKER}")
    sem_v = lambda grupo: [f for f in vars_ if f not in R.GRUPOS[grupo]]
    rankers = {
        SISTEMA: (principal, lam_sis),
        "Ablação: LambdaMART sem MMR": (principal, 0.0),
        "Ablação: sem variáveis neurais": (R.treinar(Xtr, ytr, gtr, sem_v("neural"), hiper), lam_sis),
        "Ablação: sem variáveis de preço": (R.treinar(Xtr, ytr, gtr, sem_v("preco"), hiper), lam_sis),
        "Ablação: sem variáveis relacionais": (R.treinar(Xtr, ytr, gtr, sem_v("relacionais"), hiper), lam_sis),
        "Ablação: RRF sem ranker, com MMR": (None, lam_sis),
    }

    for d in cache:
        if d["q"]["split"] != "teste":
            continue
        df = cat[d["q"]["alvo"]]
        for nome, (m, lam) in rankers.items():
            if len(d["cand"]) == 0:
                s_ = np.array([])
            else:
                s_ = m.predict(d["X"][m.features_]) if m is not None else d["X"]["rrf"].to_numpy()
            ordem = R.mmr_marcas(s_, d["marcas"], K, lam)
            linhas_b.append({**d["meta"], "metodo": nome,
                             **metricas(d["cand"][ordem], d["q"]["grades"], df)})

    a, b = pd.DataFrame(linhas_a), pd.DataFrame(linhas_b)
    a.to_csv(RES / "por_consulta_recuperacao.csv", index=False)
    b.to_csv(RES / "por_consulta_lista_final.csv", index=False)

    # ---------------- tabelas ----------------
    colsA = ["Compat@10", "Recall@100", "nDCG@10"]
    colsB = ["nDCG@10", "MRR", "Compat@10", "Diversidade@10", "Itens na lista"]
    tA = {s: a[a.split == s].groupby("metodo")[colsA].mean() for s in ("validacao", "teste")}
    for s, t in tA.items():
        t.sort_values("nDCG@10").round(3).to_csv(RES / f"tabela_recuperacao_{s}.csv")
        print(f"\n=== 1) RECUPERAÇÃO sem filtro — {s} ===\n{t.sort_values('nDCG@10').round(3).to_string()}")
    tB = b[b.split == "teste"].groupby("metodo")[colsB].mean().sort_values("nDCG@10")
    tB.round(3).to_csv(RES / "tabela_lista_final_teste.csv")
    print(f"\n=== 2) LISTA FINAL — teste ===\n{tB.round(3).to_string()}")

    gen = pd.DataFrame({"validação (plataformas do treino)": tA["validacao"]["Compat@10"],
                        "teste (plataformas novas)": tA["teste"]["Compat@10"]})
    gen["queda"] = gen.iloc[:, 0] - gen.iloc[:, 1]
    gen.round(3).to_csv(RES / "generalizacao_compat.csv")
    print(f"\n=== 3) GENERALIZAÇÃO (Compat@10 sem filtro) ===\n{gen.round(3).to_string()}")

    sock = a[a.split == "teste"].pivot_table(index="metodo", columns="socket", values="Compat@10")
    sock.round(3).to_csv(RES / "compat_por_socket_teste.csv")
    print(f"\n=== Compat@10 sem filtro por socket (teste) ===\n{sock.round(2).to_string()}")

    por_cat = b[b.split == "teste"].pivot_table(index="metodo", columns="alvo", values="nDCG@10")
    por_cat.round(3).to_csv(RES / "ndcg_por_categoria_teste.csv")

    # ---------------- confiabilidade (bootstrap pareado) ----------------
    bt = b[b.split == "teste"]
    sis = bt[bt.metodo == SISTEMA]["nDCG@10"].to_numpy()
    comps = {}
    for rival in ["Palavra-chave + filtro", "BM25 + filtro", "Neural ajustado v2 + filtro",
                  "Híbrido filtrado (RRF), sem ranker", "Ablação: sem variáveis neurais",
                  "Ablação: sem variáveis de preço", "Ablação: sem variáveis relacionais"]:
        r = bt[bt.metodo == rival]["nDCG@10"].to_numpy()
        if len(r) == len(sis):
            comps[rival] = bootstrap_dif(sis, r)
    ic = pd.DataFrame(comps, index=["Δ nDCG@10", "IC95% inf", "IC95% sup"]).T.round(3)
    ic.to_csv(RES / "comparacoes_bootstrap.csv")
    print(f"\n=== 4) Sistema − rival (nDCG@10, teste) ===\n{ic.to_string()}")

    # ---------------- importância das variáveis ----------------
    imp = pd.Series(principal.booster_.feature_importance("gain"), index=principal.features_)
    imp = imp / imp.sum()
    grupo_de = {f: g for g, fs in R.GRUPOS.items() for f in fs}
    imp_g = imp.groupby(imp.index.map(grupo_de)).sum().sort_values(ascending=False)
    imp.sort_values(ascending=False).round(4).to_csv(RES / "importancia_variaveis.csv")
    print(f"\n=== 5) Importância (ganho) por grupo de variáveis ===\n{imp_g.round(3).to_string()}")

    # ---------------- latência ponta a ponta (teste) ----------------
    rec.ranker = principal
    lat = {"embedding da consulta": [], "busca (ANN filtrada + BM25 + RRF)": [],
           "ranking (variáveis + LambdaMART + MMR)": [], "total": []}
    for q in [q for q in qs if q["split"] == "teste"][:150]:
        t1 = time.perf_counter(); rec.vetor_consulta(q["alvo"], q["montagem"]); t2 = time.perf_counter()
        c = rec.candidatos(q["alvo"], q["montagem"]); t3 = time.perf_counter()
        s, _ = rec.pontuar(q["alvo"], q["montagem"], c)
        R.mmr_marcas(s, cat[q["alvo"]]["marca"].to_numpy()[c["cand"]], K, rec.mmr); t4 = time.perf_counter()
        lat["embedding da consulta"].append((t2 - t1) * 1e3)
        lat["busca (ANN filtrada + BM25 + RRF)"].append((t3 - t2) * 1e3 - lat["embedding da consulta"][-1])
        lat["ranking (variáveis + LambdaMART + MMR)"].append((t4 - t3) * 1e3)
        lat["total"].append((t4 - t2) * 1e3)
    lat = pd.DataFrame({k: [np.mean(v), np.percentile(v, 95)] for k, v in lat.items()},
                       index=["média (ms)", "p95 (ms)"]).T.round(2)
    lat.to_csv(RES / "latencia_sistema.csv")
    print(f"\n=== 6) Latência do sistema por consulta ===\n{lat.to_string()}")

    resumo = {"consultas": grupos, "ranker": {"variaveis": melhor_lmart["variaveis"], "hiper": hiper}, "comparacoes": {k: list(v) for k, v in comps.items()},
              "ndcg_sistema_teste": float(sis.mean()),
              "importancia_grupos": imp_g.round(4).to_dict()}
    (RES / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False))

    # ---------------- gráficos ----------------
    barras(tB["nDCG@10"], RES / "ndcg_lista_final.png", "nDCG@10 (teste: AM5, LGA1700, LGA1851)",
           "Qualidade da lista final")
    barras(tA["teste"]["Compat@10"], RES / "compat_sem_filtro.png",
           "Fração compatível no top-10, SEM filtro (teste)",
           "Candidatos: o método sozinho acerta a compatibilidade?", "{:.2f}")
    sel = [m for m in ["Palavra-chave", "BM25", "Neural pré-treinado", "Neural ajustado v1",
                       "Neural ajustado v2", "Híbrido (BM25 + neural v2, RRF)"] if m in gen.index]
    g2 = gen.loc[sel]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(g2))
    ax.bar(x - 0.2, g2.iloc[:, 0], 0.4, label=g2.columns[0], color="#8a8f98")
    ax.bar(x + 0.2, g2.iloc[:, 1], 0.4, label=g2.columns[1], color=COR_SIS)
    for i, (v1, v2) in enumerate(zip(g2.iloc[:, 0], g2.iloc[:, 1])):
        ax.text(i - 0.2, v1 + 0.01, f"{v1:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.2, v2 + 0.01, f"{v2:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, [s.replace(" (BM25 + neural v2, RRF)", "\n(BM25+v2)").replace("Neural ", "Neural\n")
                      for s in g2.index], fontsize=8.5)
    ax.set_ylabel("Compat@10 sem filtro"); ax.set_ylim(0, 1.08)
    ax.set_title("Generalização: plataformas vistas no treino x plataformas novas")
    ax.legend(loc="upper left", fontsize=8.5); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(RES / "generalizacao.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(imp_g.index[::-1], imp_g.values[::-1], color=COR_SIS)
    for i, v in enumerate(imp_g.values[::-1]):
        ax.text(v + 0.005, i, f"{v:.0%}", va="center", fontsize=9)
    ax.set_xlabel("Participação no ganho total do LambdaMART")
    ax.set_title("O que o ranking usa para ordenar")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(RES / "importancia_variaveis.png", dpi=150); plt.close(fig)
    print(f"\nConcluído em {time.time()-t0:.0f}s. Resultados em {RES}")


if __name__ == "__main__":
    main()
