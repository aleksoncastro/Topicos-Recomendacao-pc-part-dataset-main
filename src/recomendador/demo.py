"""Demonstração: escolha uma CPU e o sistema monta o resto do PC.

Uso:
    python -m src.recomendador.demo "Ryzen 5 5600"
    python -m src.recomendador.demo "Core i5-14600K" --n 3
    python -m src.recomendador.demo --busca "processador AMD barato para jogos"
"""
import argparse
import sys
import time

from .dados import NOME_PT
from .recomendador import Recomendador

sys.stdout.reconfigure(encoding="utf-8")


def preco(v):
    return f"US$ {v:8.2f}" if v == v else "   s/ preço"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cpu", nargs="?", default="AMD Ryzen 5 5600")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--busca", help="busca livre por CPU em linguagem natural")
    a = ap.parse_args()

    print("Carregando catálogo e índices...")
    rec = Recomendador()

    if a.busca:
        print(f'\nBusca: "{a.busca}"')
        for _, r in rec.buscar_texto(a.busca, "cpu", a.n).iterrows():
            print(f"  {r['similaridade']:.3f}  {r['name']:<45} {preco(r['price'])}  {r['socket']}")
        return

    cpu = rec.encontrar("cpu", a.cpu)
    print(f"\nCPU escolhida: {cpu['name']} | {cpu['microarchitecture']} | socket {cpu['socket']} "
          f"| {cpu['tdp']:.0f} W | {preco(cpu['price'])}")
    rec.recomendar("motherboard", {"cpu": cpu}, 1)  # aquece o modelo (1ª carga não entra no tempo)
    t = time.perf_counter()
    montagem, passos = rec.montar(cpu, n=a.n)
    dt = (time.perf_counter() - t) * 1000

    for alvo, recs in passos:
        print(f"\n▶ {NOME_PT[alvo].upper()} — {a.n} recomendações compatíveis")
        for i, (_, r) in enumerate(recs.iterrows()):
            marca = "★" if i == 0 else " "
            print(f"  {marca} {r['name'][:50]:<50} {preco(r['price'])}  sim={r['similaridade']:.3f}")

    total = sum(r["price"] for r in montagem.values() if r["price"] == r["price"])
    print("\n=== MONTAGEM FINAL (1ª recomendação de cada etapa) ===")
    for c, r in montagem.items():
        print(f"  {NOME_PT[c]:<15} {r['name'][:50]:<50} {preco(r['price'])}")
    print(f"  {'TOTAL':<15} {'':<50} {preco(total)}")
    print(f"\nTempo para montar o PC inteiro: {dt:.0f} ms")


if __name__ == "__main__":
    main()
