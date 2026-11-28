"""Gera docs/Plano_e_Progresso.pdf — documentação do plano e do que já foi feito."""
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

AQUI = Path(__file__).resolve().parent
RES = AQUI.parent / "resultados"
SAIDA = AQUI / "Plano_e_Progresso.pdf"

F = "C:/Windows/Fonts/"
pdfmetrics.registerFont(TTFont("Arial", F + "arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-B", F + "arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Arial-I", F + "ariali.ttf"))
pdfmetrics.registerFont(TTFont("Arial-BI", F + "arialbi.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("Arial", normal="Arial", bold="Arial-B", italic="Arial-I", boldItalic="Arial-BI")

LARANJA = colors.HexColor("#B4581A")
AZUL = colors.HexColor("#1B2440")
CINZA = colors.HexColor("#5A6270")
CLARO = colors.HexColor("#F6EFE8")
BORDA = colors.HexColor("#D9D2C7")

ss = getSampleStyleSheet()
N = ParagraphStyle("n", parent=ss["Normal"], fontName="Arial", fontSize=10, leading=14.5,
                   textColor=colors.HexColor("#222"), spaceAfter=6)
P = ParagraphStyle("p", parent=N, spaceAfter=3)
H1 = ParagraphStyle("h1", parent=N, fontName="Arial-B", fontSize=16, leading=20, textColor=AZUL,
                    spaceBefore=10, spaceAfter=8)
H2 = ParagraphStyle("h2", parent=N, fontName="Arial-B", fontSize=12, leading=16, textColor=LARANJA,
                    spaceBefore=8, spaceAfter=4)
TAG = ParagraphStyle("tag", parent=N, fontName="Arial-B", fontSize=8.5, textColor=LARANJA, spaceAfter=2)
CEL = ParagraphStyle("cel", parent=N, fontSize=8.8, leading=11.5, spaceAfter=0)
CELB = ParagraphStyle("celb", parent=CEL, fontName="Arial-B", textColor=AZUL)
NOTA = ParagraphStyle("nota", parent=N, fontName="Arial-I", fontSize=8.5, leading=11, textColor=CINZA)
CAIXA = ParagraphStyle("caixa", parent=N, fontSize=9.5, leading=13.5, spaceAfter=0)


def b(t):  # bullet
    return Paragraph(t, ParagraphStyle("b", parent=P, leftIndent=14, bulletIndent=4), bulletText="•")


def tabela(linhas, larguras, destaque=None, cab=True):
    dados = [[Paragraph(str(c), CELB if (cab and i == 0) else CEL) for c in l]
             for i, l in enumerate(linhas)]
    t = Table(dados, colWidths=larguras, repeatRows=1 if cab else 0)
    est = [("GRID", (0, 0), (-1, -1), 0.4, BORDA),
           ("VALIGN", (0, 0), (-1, -1), "TOP"),
           ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
           ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]
    if cab:
        est.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ECE7DF")))
    for r in destaque or []:
        est.append(("BACKGROUND", (0, r), (-1, r), CLARO))
    t.setStyle(TableStyle(est))
    return t


def caixa(texto, fundo=CLARO, borda=LARANJA):
    t = Table([[Paragraph(texto, CAIXA)]], colWidths=[17 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), fundo),
                           ("BOX", (0, 0), (-1, -1), 0.8, borda),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return t


def diagrama():
    d = Drawing(17 * cm, 4.6 * cm)
    caixas = [
        ("Montagem atual", "CPU escolhida +\npeças já definidas", False),
        ("1. Candidatos", "Embedding da montagem\n→ busca ANN (HNSW)\ntop-100 da categoria", True),
        ("2. Ranking", "Filtro de compatibilidade\n+ similaridade\n+ popularidade, preço\n+ diversidade (MMR)", True),
        ("Lista final", "Top-N peças\ncompatíveis", False),
    ]
    w, h, gap = 3.75 * cm, 3.3 * cm, 0.68 * cm
    y = 1.0 * cm
    for i, (tit, sub, dest) in enumerate(caixas):
        x = i * (w + gap)
        d.add(Rect(x, y, w, h, rx=6, ry=6, fillColor=CLARO if dest else colors.white,
                   strokeColor=LARANJA if dest else BORDA, strokeWidth=1))
        d.add(String(x + 8, y + h - 16, tit, fontName="Arial-B", fontSize=10, fillColor=AZUL))
        for j, linha in enumerate(sub.split("\n")):
            d.add(String(x + 8, y + h - 33 - j * 11.5, linha, fontName="Arial", fontSize=7.8,
                         fillColor=CINZA))
        if i < len(caixas) - 1:
            ax = x + w + 4
            d.add(Line(ax, y + h / 2, ax + gap - 10, y + h / 2, strokeColor=CINZA, strokeWidth=1.4))
            d.add(Polygon([ax + gap - 10, y + h / 2 + 4, ax + gap - 3, y + h / 2,
                           ax + gap - 10, y + h / 2 - 4], fillColor=CINZA, strokeColor=CINZA))
    d.add(String(0, 0.35 * cm, "A peça escolhida volta para a montagem e o ciclo se repete para a próxima categoria:",
                 fontName="Arial-I", fontSize=7.8, fillColor=CINZA))
    d.add(String(0, 0.0 * cm, "placa-mãe → memória → GPU → cooler → fonte → gabinete → armazenamento.",
                 fontName="Arial-I", fontSize=7.8, fillColor=CINZA))
    return d


def rodape(canvas, doc):
    canvas.saveState()
    canvas.setFont("Arial", 8)
    canvas.setFillColor(CINZA)
    canvas.drawString(2 * cm, 1.2 * cm, "Sistemas de Recomendação · Projeto Final · Plano e progresso")
    canvas.drawRightString(19 * cm, 1.2 * cm, f"{doc.page}")
    canvas.restoreState()


def capa(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(LARANJA)
    canvas.rect(0, A4[1] - 0.6 * cm, A4[0], 0.6 * cm, fill=1, stroke=0)
    canvas.restoreState()
    rodape(canvas, doc)


s = []
# ---------------------------------------------------------------- capa
s += [Spacer(1, 2.2 * cm),
      Paragraph("PROJETO FINAL · SISTEMAS DE RECOMENDAÇÃO", TAG),
      Paragraph("Recomendador de peças de PC compatíveis com Neural Search",
                ParagraphStyle("t", parent=H1, fontSize=24, leading=29)),
      Paragraph("Documentação do plano e do progresso até 26/11/2026 · Apresentação em 04/12",
                ParagraphStyle("st", parent=N, fontSize=11, textColor=CINZA)),
      Spacer(1, 0.6 * cm),
      caixa("<b>Em uma frase:</b> o usuário escolhe uma peça (por exemplo, um <b>AMD Ryzen 5 5600</b>) e o sistema "
            "recomenda, passo a passo, as demais peças compatíveis (placa-mãe AM4, memória DDR4, fonte com "
            "potência suficiente, gabinete do tamanho certo…), como os sites de montagem de PC — usando "
            "embeddings, índice vetorial (ANN) e ranqueamento em duas etapas."),
      Spacer(1, 0.6 * cm),
      Paragraph("Sumário", H2)]
for item in ["1. Requisitos do trabalho (A2.pdf)", "2. Caso de uso e pergunta base",
             "3. Dados e regras de compatibilidade", "4. Arquitetura do sistema",
             "5. Relevância, divisão temporal, baselines e métricas", "6. O que já foi feito",
             "7. Resultados preliminares (rodada 1)", "8. Ameaças à validade e teste A/B",
             "9. Próximos passos e cronograma", "10. Como executar"]:
    s.append(Paragraph(item, P))
s.append(PageBreak())

# ---------------------------------------------------------------- 1
s += [Paragraph("1. Requisitos do trabalho (A2.pdf)", H1),
      Paragraph("O enunciado pede um <b>caso de uso de recomendação baseado em Neural Search</b>, avaliado em "
                "múltiplas camadas: <b>embeddings, índice vetorial, geração de candidatos, ranking e qualidade "
                "da lista</b>. Entrega: <b>relatório + código + apresentação</b> (04/12).", N),
      Paragraph("Perguntas que norteiam a avaliação", H2),
      b("<b>P1.</b> A busca neural recupera itens mais relevantes do que as alternativas simples?"),
      b("<b>P2.</b> Quanto se perde ao usar busca aproximada (ANN) no lugar da exata, e quanto se ganha em velocidade?"),
      b("<b>P3.</b> Os resultados são confiáveis ou podem estar inflados por vazamento de dados, viés ou métrica mal escolhida?"),
      Paragraph("Estrutura mínima do relatório e onde cada item é atendido", H2),
      tabela([["Seção exigida", "Como este projeto atende"],
              ["1. Caso de uso e pergunta base", "Montagem de PC assistida: recomendar peças compatíveis a partir de uma peça inicial (seção 2)."],
              ["2. Dados, relevância e divisão temporal", "Catálogo PCPartPicker limpo; relevância graduada 0/1/2; divisão por geração de plataforma (seções 3 e 5)."],
              ["3. Baselines e modelo neural", "Popularidade, palavra-chave, BM25 vs. neural pré-treinado e ajustado; candidatos + ranking (seções 4 e 5)."],
              ["4. Busca exata vs. ANN", "FAISS Flat vs. HNSW, IVF e IVF-PQ: ANN Recall@10, latência, memória; 63 mil e 500 mil itens."],
              ["5. Matriz de avaliação", "Recuperação (Compat@10, Recall@100), ranking (nDCG@10, MRR), lista (diversidade, tamanho)."],
              ["6. Benefícios", "Ablações: tirar similaridade neural, tirar coerência de preço, tirar o ranking."],
              ["7. Ameaças e teste A/B", "Circularidade da relevância, dados sintéticos, mapeamentos por regra; proposta A/B (seção 8)."],
              ["8. Conclusão", "Resposta à pergunta base para este caso, a partir dos resultados finais."]],
             [5.2 * cm, 11.8 * cm]),
      PageBreak()]

# ---------------------------------------------------------------- 2
s += [Paragraph("2. Caso de uso e pergunta base", H1),
      Paragraph("O dataset é um <b>catálogo</b> (63.827 peças, 25 categorias) sem usuários, cliques ou datas. Por isso o "
                "caso de uso escolhido é a <b>recomendação a partir de um contexto</b>: a montagem em andamento funciona "
                "como a “sessão” do usuário (slide 12: <i>contexto da sessão</i>).", N),
      caixa("<b>Pergunta base aplicada:</b> dado um PC em montagem (por exemplo, só a CPU escolhida), como escolher "
            "as ~10 peças mais adequadas de cada categoria entre milhares de opções — compatíveis, na faixa de "
            "preço coerente e variadas — rápido o bastante para o usuário não esperar?"),
      Spacer(1, 6),
      Paragraph("Exemplo de fluxo", H2),
      tabela([["Passo", "Contexto", "O sistema recomenda"],
              ["1", "Ryzen 5 5600 (Zen 3, 65 W)", "Placas-mãe AM4 / DDR4 (B550, A520) de faixa intermediária"],
              ["2", "+ placa-mãe escolhida", "Kits DDR4 que cabem nos slots e na memória máxima da placa"],
              ["3", "+ GPU escolhida", "Fontes com potência ≥ (TDP CPU + consumo GPU + 75 W) × 1,3"],
              ["4", "+ placa-mãe", "Gabinetes que comportam o formato da placa (ITX, mATX, ATX, EATX)"]],
             [1.4 * cm, 5 * cm, 10.6 * cm]),
      PageBreak()]

# ---------------------------------------------------------------- 3
s += [Paragraph("3. Dados e regras de compatibilidade", H1),
      Paragraph("Base: saída do pipeline de limpeza já existente (<font face='Arial-I'>pc_parts_cleaning/data/processed</font>). "
                "O recomendador usa as 8 categorias de uma montagem, apenas itens <b>com preço</b> "
                "(~10,3 mil itens). O benchmark de ANN usa o catálogo completo (63,8 mil).", N),
      tabela([["Categoria", "Itens c/ preço", "Atributos usados"],
              ["CPU", "543", "microarquitetura → socket, TDP, núcleos, GPU integrada"],
              ["Placa-mãe", "951", "socket, formato, slots, memória máx., DDR (derivado)"],
              ["Memória", "2.884", "versão DDR, MHz, nº de módulos, GB total"],
              ["Placa de vídeo", "1.273", "chipset → consumo estimado (W), VRAM, comprimento"],
              ["Fonte", "566", "potência, formato (ATX/SFX), eficiência"],
              ["Gabinete", "1.053", "tipo → maior placa suportada"],
              ["Cooler", "945", "radiador (AIO) ou ar, ruído"],
              ["Armazenamento", "2.095", "SSD/HDD, capacidade, interface"]],
             [3.2 * cm, 2.6 * cm, 11.2 * cm]),
      Spacer(1, 6),
      Paragraph("Atributos derivados (o dataset não traz pronto)", H2),
      b("<b>Socket da CPU:</b> tabela microarquitetura → socket (Zen 3 → AM4, Zen 4/5 → AM5, Alder/Raptor Lake → LGA1700, "
        "Arrow Lake → LGA1851…), com exceções pelo nome (linhas HEDT X/XE, Xeon E5, FX vs. APU). Threadripper/EPYC ficam fora."),
      b("<b>DDR da placa-mãe:</b> pelo socket (AM5/LGA1851 → DDR5; AM4/LGA1200 → DDR4); no LGA1700, “DDR4”/“D4” no nome."),
      b("<b>Consumo da GPU:</b> tabela aproximada de TBP por chipset (ex.: RTX 4070 ≈ 200 W); padrão 200 W."),
      b("<b>Popularidade:</b> a ordem original do scraper segue a ordenação padrão do PCPartPicker; usada como ranking de popularidade."),
      Paragraph("Regras de compatibilidade (restrições do negócio)", H2),
      tabela([["Alvo", "Regra"],
              ["Placa-mãe", "socket = socket da CPU; DDR compatível; cabe no gabinete; aceita a memória escolhida"],
              ["Memória", "DDR = DDR da placa; GB total ≤ memória máx.; nº módulos ≤ slots"],
              ["Gabinete", "maior placa suportada ≥ formato da placa; Mini ITX exige fonte SFX"],
              ["Fonte", "potência ≥ (TDP + W da GPU + 75) × 1,3"],
              ["Cooler", "cooler box Intel/AMD só para CPU da mesma marca"]],
             [3.2 * cm, 13.8 * cm]),
      Paragraph("Limitações do dataset: não há socket dos coolers, comprimento máximo de GPU no gabinete nem datas de lançamento.", NOTA),
      PageBreak()]

# ---------------------------------------------------------------- 4
s += [Paragraph("4. Arquitetura do sistema", H1),
      Paragraph("Segue a arquitetura em duas etapas vista em aula (slides 4, 11–13).", N),
      diagrama(), Spacer(1, 8),
      tabela([["Etapa", "O que faz", "Implementação"],
              ["Representação", "Cada peça vira um texto descritivo (nome, marca, especificações, socket, DDR, faixa de preço) e depois um vetor.",
               "Modelo multilíngue <i>paraphrase-multilingual-MiniLM-L12-v2</i> (384 dim.); aceita consultas em português."],
              ["Fine-tuning", "Ensina o modelo que “montagem X” fica perto de peças compatíveis e longe das incompatíveis.",
               "Aprendizado contrastivo (MultipleNegativesRankingLoss) com negativos difíceis, só nas plataformas antigas."],
              ["1. Candidatos", "A montagem atual vira um vetor de consulta; busca os 100 vizinhos na categoria pedida.",
               "FAISS HNSW (M=32); prioridade: velocidade e cobertura."],
              ["2. Ranking", "Remove incompatíveis e ordena por sinais detalhados.",
               "Score = similaridade + popularidade + coerência de preço − penalidade de outlier; MMR por marca para diversidade. Pesos ajustados na validação."],
              ["Montagem sequencial", "A peça escolhida entra no contexto da próxima recomendação.",
               "<i>Recomendador.montar()</i>; PC completo em ~0,24 s."]],
             [2.8 * cm, 6.6 * cm, 7.6 * cm]),
      PageBreak()]

# ---------------------------------------------------------------- 5
s += [Paragraph("5. Relevância, divisão temporal, baselines e métricas", H1),
      Paragraph("Definição de relevância (graduada)", H2),
      tabela([["Nota", "Significado"],
              ["0", "Incompatível com a montagem"],
              ["1", "Compatível"],
              ["2", "Compatível <b>e</b> na mesma faixa de preço: |percentil do item entre os compatíveis − percentil da CPU no seu socket| ≤ 0,25"]],
             [1.6 * cm, 15.4 * cm]),
      Paragraph("Consultas de avaliação: para cada CPU com preço das plataformas estudadas, 4 consultas "
                "(placa-mãe | CPU; memória | CPU + placa; gabinete | CPU + placa; fonte | CPU + GPU de faixa parecida). "
                "Total: <b>1.044 consultas</b>.", N),
      Paragraph("Divisão “temporal”", H2),
      Paragraph("Sem datas no dataset, usa-se a <b>geração da plataforma</b> como eixo do tempo: o sistema é ajustado "
                "em plataformas antigas e testado em plataformas novas que ele não viu no treino.", N),
      tabela([["Grupo", "Plataformas", "Uso"],
              ["Ajuste (70% das CPUs antigas)", "LGA1151, AM4, LGA1200 (até 2020)", "Fine-tuning dos embeddings"],
              ["Validação (30% das CPUs antigas)", "LGA1151, AM4, LGA1200", "Escolha dos pesos do ranking"],
              ["Teste", "LGA1700, AM5, LGA1851 (2021+)", "Números reportados"]],
             [5 * cm, 6.2 * cm, 5.8 * cm]),
      Paragraph("Baselines", H2),
      b("<b>Popularidade:</b> os mais populares da categoria."),
      b("<b>Palavra-chave:</b> conta termos em comum entre a montagem e o item."),
      b("<b>BM25:</b> busca lexical clássica (TF-IDF aprimorado)."),
      b("<b>Neural pré-treinado</b> (texto bruto e enriquecido) e <b>neural ajustado</b> (fine-tuning)."),
      Paragraph("Todos são avaliados <b>sem filtro</b> (o método sozinho acerta a compatibilidade?) e <b>com filtro</b> (lista final justa).", N),
      Paragraph("Matriz de avaliação", H2),
      tabela([["Camada", "Métrica", "O que responde"],
              ["Recuperação", "Compat@10, Recall@100", "Os candidatos trazem itens compatíveis?"],
              ["Ranking", "nDCG@10, MRR", "Os melhores itens estão no topo?"],
              ["Qualidade da lista", "Diversidade@10 (marcas), itens na lista", "A lista é variada e completa?"],
              ["Índice", "ANN Recall@10, latência média/p95, tamanho", "Quanto se perde e ganha com ANN? (P2)"],
              ["Confiabilidade", "IC 95% por bootstrap pareado; ablações", "O ganho é real? (P3)"]],
             [3.4 * cm, 6 * cm, 7.6 * cm]),
      PageBreak()]

# ---------------------------------------------------------------- 6
s += [Paragraph("6. O que já foi feito", H1),
      Paragraph("Código em <font face='Arial-I'>recomendador/</font> (Python 3.12, ambiente próprio em <font face='Arial-I'>.venv</font>).", N),
      tabela([["Arquivo", "Conteúdo", "Status"],
              ["src/dados.py", "Carga do catálogo, socket/DDR/formatos/consumo derivados, popularidade, faixas de preço", "Pronto"],
              ["src/compat.py", "Regras de compatibilidade e relevância graduada 0/1/2", "Pronto"],
              ["src/textos.py", "Textos “bruto” e “enriquecido” das peças e da montagem", "Pronto"],
              ["src/embeddings.py", "Embeddings com cache; índices Flat, HNSW, IVF, IVF-PQ", "Pronto"],
              ["src/recomendador.py", "Candidatos + ranking + MMR + montagem sequencial + busca em português", "Pronto"],
              ["src/baselines.py", "Popularidade, palavra-chave, BM25", "Pronto"],
              ["src/avaliacao.py", "Consultas, métricas, ajuste de pesos, ablações, bootstrap, gráficos", "Rodada 1 concluída"],
              ["src/finetune.py", "Fine-tuning contrastivo com negativos difíceis", "Em execução"],
              ["src/ann_benchmark.py", "Exata vs. ANN (63 mil e 500 mil itens)", "Escrito, falta rodar"],
              ["demo.py", "Demonstração: CPU → PC completo no terminal", "Pronto"]],
             [3.6 * cm, 10.4 * cm, 3 * cm], destaque=[8, 9]),
      Spacer(1, 8),
      Paragraph("Correções já feitas após o primeiro teste da demo", H2),
      b("Busca por nome “Ryzen 5 5600” casava com “5600X” — agora prioriza o nome exato."),
      b("Cooler box da Intel sugerido para CPU AMD — nova regra de compatibilidade."),
      b("Recomendação de GPU fora da faixa e enviesada pela marca da CPU — a ser resolvida pelos pesos ajustados e pela busca híbrida."),
      PageBreak()]

# ---------------------------------------------------------------- 7
s += [Paragraph("7. Resultados preliminares (rodada 1 — modelo pré-treinado, sem fine-tuning)", H1),
      Paragraph("Conjunto de teste (plataformas novas, 344 consultas). Estes números servem de linha de base e "
                "serão refeitos com o modelo ajustado.", NOTA),
      Paragraph("Recuperação sem filtro de compatibilidade", H2),
      tabela([["Método", "Compat@10", "Recall@100", "nDCG@10"],
              ["Palavra-chave", "0,937", "0,182", "0,749"],
              ["BM25", "0,625", "0,169", "0,451"],
              ["Popularidade", "0,678", "0,114", "0,435"],
              ["Neural pré-treinado (texto enriquecido)", "0,653", "0,117", "0,429"],
              ["Neural pré-treinado (texto bruto)", "0,472", "0,103", "0,299"]],
             [8.2 * cm, 2.9 * cm, 2.9 * cm, 3 * cm]),
      Spacer(1, 4)]
if (RES / "compat_sem_filtro.png").exists():
    s.append(Image(str(RES / "compat_sem_filtro.png"), width=14 * cm, height=7 * cm))
s += [Paragraph("Lista final (com filtro de compatibilidade)", H2),
      tabela([["Método", "nDCG@10", "MRR", "Diversidade@10"],
              ["Sistema: neural + ranking", "0,856", "0,861", "0,444"],
              ["Palavra-chave + filtro", "0,800", "0,783", "0,385"],
              ["BM25 + filtro", "0,730", "0,753", "0,349"],
              ["Neural pré-treinado + filtro", "0,647", "0,660", "0,331"],
              ["Popularidade + filtro", "0,626", "0,567", "0,424"],
              ["Ablação: ranking sem similaridade neural", "0,856", "0,861", "0,444"]],
             [8.2 * cm, 2.9 * cm, 2.9 * cm, 3 * cm], destaque=[1, 6]),
      Paragraph("Diferença sistema − melhor baseline (palavra-chave + filtro): Δ nDCG@10 = +0,056 (IC 95%: +0,033 a +0,080). "
                "Latência da busca HNSW: 0,14 ms por consulta; a codificação do texto da consulta pelo modelo domina (~23 ms).", NOTA),
      Spacer(1, 4),
      KeepTogether([Paragraph("Leitura honesta destes resultados", H2),
                    caixa("<b>1.</b> O modelo pré-treinado <b>não entende compatibilidade</b>: sem filtro, só 65% do seu top-10 é compatível, "
                          "contra 94% da palavra-chave — termos exatos como “AM4” e “DDR5” decidem a compatibilidade.<br/><br/>"
                          "<b>2.</b> O ganho do sistema veio do <b>ranking</b>, não do neural: o ajuste escolheu peso zero para a "
                          "similaridade (a ablação “sem similaridade neural” empata com o sistema).<br/><br/>"
                          "<b>3.</b> Há <b>circularidade</b>: a nota 2 da relevância depende de faixa de preço e o ranking usa "
                          "coerência de preço como sinal. Isso infla o nDCG do sistema (ameaça à validade, P3).<br/><br/>"
                          "<b>Resposta:</b> fine-tuning do modelo (em execução), busca híbrida e avaliação também por métricas "
                          "que não dependem de preço (Compat@10 sem filtro, Recall@100).")]),
      PageBreak()]

# ---------------------------------------------------------------- 8
s += [Paragraph("8. Ameaças à validade e proposta de teste A/B", H1),
      tabela([["Ameaça", "Risco", "Mitigação"],
              ["Relevância sintética", "Notas criadas por regras, não por usuários reais.", "Declarar; validar algumas montagens manualmente; teste A/B."],
              ["Circularidade (preço)", "Ranking e relevância usam faixa de preço.", "Reportar ablação sem coerência e métricas só de compatibilidade."],
              ["Circularidade (regras)", "Filtro e relevância usam as mesmas regras.", "Avaliar candidatos ANTES do filtro (Compat@10, Recall@100)."],
              ["Vazamento", "Fine-tuning ver as plataformas de teste.", "Ajuste só em CPUs antigas; pesos na validação; teste em plataformas novas."],
              ["Mapeamentos por regra", "Socket/DDR/consumo derivados podem errar.", "Tabelas documentadas; exceções tratadas pelo nome."],
              ["Popularidade aproximada", "Ordem do scraper como proxy.", "Declarar; testar o sistema sem esse sinal."],
              ["Preços", "Retrato de um momento, em USD.", "Faixas relativas (percentis), não valores absolutos."]],
             [3.6 * cm, 6 * cm, 7.4 * cm]),
      Paragraph("Proposta de teste A/B", H2),
      tabela([["Item", "Definição"],
              ["Grupos", "A: filtro de compatibilidade + popularidade (como sites atuais). B: sistema neural híbrido + ranking."],
              ["Unidade", "Usuário (sessão de montagem), divisão aleatória 50/50."],
              ["Métrica principal", "Taxa de montagens concluídas (todas as peças adicionadas)."],
              ["Secundárias", "Cliques nas recomendações, posição do item escolhido, valor da montagem."],
              ["Guarda-corpo", "Latência p95 < 200 ms; taxa de incompatibilidade reportada = 0."],
              ["Duração", "≥ 2 semanas (cobrir dias úteis e fins de semana); tamanho de amostra por cálculo de poder."]],
             [3.4 * cm, 13.6 * cm]),
      PageBreak()]

# ---------------------------------------------------------------- 9
s += [Paragraph("9. Próximos passos e cronograma", H1),
      Paragraph("Para tornar o sistema mais robusto (e mais completo do que um filtro com ordenação):", N),
      tabela([["#", "Passo", "Por quê"],
              ["1", "Concluir o fine-tuning e refazer a avaliação", "Medir se o modelo aprende compatibilidade e generaliza para AM5/LGA1700/LGA1851 (P1)."],
              ["2", "Busca híbrida: BM25 + neural com Reciprocal Rank Fusion", "Slide 9 (hibridização): termos exatos + significado; corrige a fraqueza do neural."],
              ["3", "Ranking aprendido (LightGBM LambdaMART)", "Substitui a soma ponderada; importância das variáveis explica o ganho (seção 6) e reduz a caixa-preta."],
              ["4", "Rodar o benchmark exata vs. ANN", "Curvas recall × latência em 63 mil e 500 mil itens (P2)."],
              ["5", "Notebook de apresentação", "Demo ao vivo + tabelas + gráficos."],
              ["6", "Relatório (8 seções) e slides", "Entrega final."]],
             [0.8 * cm, 6.6 * cm, 9.6 * cm]),
      Spacer(1, 6),
      tabela([["Data", "Atividade"],
              ["26–27/11", "Fine-tuning, avaliação com modelo ajustado, benchmark ANN"],
              ["28–29/11", "Busca híbrida (RRF) e ranking LambdaMART; nova rodada de avaliação"],
              ["30/11–01/12", "Notebook final, gráficos e tabelas"],
              ["02–03/12", "Relatório e slides; ensaio da apresentação"],
              ["04/12", "Apresentação"]],
             [3.4 * cm, 13.6 * cm]),
      Spacer(1, 10),
      Paragraph("10. Como executar", H1),
      tabela([["Comando (na pasta recomendador/)", "Efeito"],
              [".venv\\Scripts\\python -m src.finetune", "Treina o modelo ajustado (modelos/minilm-pc-compat)"],
              [".venv\\Scripts\\python -m src.avaliacao", "Gera tabelas, pesos e gráficos em resultados/"],
              [".venv\\Scripts\\python -m src.ann_benchmark", "Benchmark exata vs. ANN"],
              [".venv\\Scripts\\python demo.py \"Ryzen 5 5600\"", "Monta um PC a partir da CPU"],
              [".venv\\Scripts\\python demo.py --busca \"processador barato para jogos\"", "Busca livre em português"]],
             [8.6 * cm, 8.4 * cm])]

doc = SimpleDocTemplate(str(SAIDA), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=1.8 * cm, bottomMargin=2 * cm,
                        title="Plano e progresso — Recomendador de peças de PC",
                        author="Projeto Final de Sistemas de Recomendação")
doc.build(s, onFirstPage=capa, onLaterPages=rodape)
print("PDF gerado:", SAIDA)
