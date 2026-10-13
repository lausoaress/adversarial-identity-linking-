# adversarial-identity-linking-

Adversarial multi-agent verification (Proposer–Contester–Judge) for reducing
false positives in dark web ↔ open web identity linking — undergraduate
thesis (UFSC).

Protótipo de TCC: um pipeline Proposer → Contestador → Juiz para reduzir
falsos positivos na vinculação de identidade entre aliases da dark web e
contas da internet aberta, comparado contra o baseline de agente único de
[Arabnezhad et al. (2020), "A Light in the Dark Web"](https://gwern.net/doc/darknet-market/2020-arabnezhad.pdf).

O objetivo concreto que este código mede: dos pares candidatos que o
baseline deixa como **"Unclear"** no experimento real deles (43% dos casos
Reddit ↔ dark web — ver `docs/` ou a seção "Trabalhos Relacionados" do
artigo), quantos a camada adversarial consegue resolver com uma decisão
auditável?

## Status atual

O que já está implementado e testado (ver "Rodando" abaixo):

- **Extração de features** (`src/data/features.py`) — reproduz exatamente
  a especificação da Arabnezhad et al.: n-gramas de palavra (1-3) e
  caractere (1-5), frequência de pontuação/dígitos/caracteres especiais, e
  o fingerprint de atividade diária `P_u[h]`.
- **Baseline de dois estágios** (`src/baseline/cosine_pipeline.py`) — a
  mesma pipeline k-atribuição → decisão final por cosseno (k=10, limiar
  0,4190) do artigo principal.
- **Construção de ground truth sintético** (`src/data/alter_ego.py`) — a
  metodologia de "alter-ego" deles (dividir as mensagens de um usuário em
  dois conjuntos), que **não depende do dataset proprietário da
  Arabnezhad** — funciona sobre qualquer corpus que você carregar.
- **Os três agentes** (`src/agents/`) — Proposer, Contestador e Juiz, com
  os prompts já alinhados às escolhas provisórias do artigo (Seção 3) e
  ao achado central da revisão: o Contestador prioriza explicitamente
  buscar **contradição factual declarada** (idade, localização, religião,
  política, hábitos) antes de dúvida estilística, porque foi exatamente
  esse tipo de evidência que explicou os falsos positivos reais no artigo
  da Arabnezhad.
- **Orquestração sequencial** (`src/agents/orchestrator.py`) — uma rodada
  por enquanto; já preparado para virar um grafo LangGraph multi-rodada
  depois (ver o docstring do arquivo).
- **Métricas de comparação** (`src/evaluation/metrics.py` +
  `run_experiment.py`) — precisão, recall, taxa de falso positivo e,
  principalmente, a taxa de resolução de casos "Unclear".
- **Testes** (`tests/`) e um **demo sem necessidade de modelo**
  (`scripts/run_demo.py`), usando um LLM "stub" com respostas roteirizadas
  — confirma que toda a pipeline conecta antes de gastar tempo com um
  modelo de verdade.

O que **ainda não está feito** (próximos passos, nessa ordem):

1. Acesso a dados reais — ver `data/README.md` (decisão: Gwern/The Majestic
   Garden para o lado dark web, Reddit para o lado internet aberta).
2. Rodar contra um modelo de verdade (local via Ollama) em vez do stub.
3. Calibrar os pesos `stylometric_weight` / `temporal_weight` e validar o
   protocolo estatístico (Juola, 2021) sobre um conjunto rotulado.
4. Multi-rodada no Juiz (Han et al., 2025) — hoje é uma rodada só.

## Rodando

Requer Python 3.10+.

```bash
pip install -r requirements.txt

# 1. Confirma que a pipeline inteira conecta, sem precisar de modelo nenhum:
python scripts/run_demo.py

# 2. Roda os testes:
pytest tests/ -v

# 3. Experimento completo (baseline vs. adversarial) com dados sintéticos
#    e sem modelo (mesma ideia do demo, mas em volume maior):
python -m src.evaluation.run_experiment --provider stub --n-users 20
```

### Rodando com um modelo local (Ollama)

Para validar com modelo local primeiro, antes de gastar com API:

```bash
./scripts/setup_ollama.sh llama3.1   # baixa o modelo (troque pelo que preferir)
ollama serve                          # em outro terminal, mantém rodando

python -m src.evaluation.run_experiment --provider ollama --model llama3.1
```

Trocar de provedor depois (OpenAI, Anthropic) é só mudar `--provider` (ou
`llm.provider` em `config/config.yaml`) — nenhum agente precisa mudar,
porque todos falam só com a interface `LLMClient`
(`src/agents/llm_client.py`).

## Estrutura

```
src/
  schemas.py              # Alias, CandidatePair, Evidence, Verdict, PipelineResult, ...
  data/
    loaders.py             # carregadores reais (a preencher) + gerador sintético
    alter_ego.py            # construção de ground truth sintético
    features.py              # extração de features (espec. exata da Arabnezhad et al.)
  baseline/
    cosine_pipeline.py        # o baseline de agente único — não mexer; é o alvo de comparação
  agents/
    llm_client.py               # abstração de provedor (Ollama / OpenAI / Anthropic / stub)
    proposer.py                  # gera a hipótese de vínculo
    contestador.py                 # busca contradição factual e explicações alternativas
    juiz.py                         # pondera e decide, com explicação citável
    orchestrator.py                  # encadeia os três
  evaluation/
    metrics.py                        # precisão, recall, taxa de FP, taxa de "Unclear" resolvido
    run_experiment.py                  # ponto de entrada: baseline vs. adversarial
scripts/
  setup_ollama.sh          # baixa o modelo local
  run_demo.py              # roda 1 par, sem modelo, pra conferir que tudo conecta
tests/
  test_features.py         # extração de features
  test_agents_smoke.py     # prompts/parsing dos 3 agentes, com LLM "stub"
config/config.yaml         # todos os parâmetros num lugar só
data/
  README.md                # decisão de datasets + de onde vêm os dados
  raw/                      # corpora brutos (git-ignorado)
  processed/                 # corpora pós-processamento (git-ignorado)
docs/
  figures/                 # imagens para o artigo/README (diagramas, gráficos de resultado)
```

## Como isso se liga ao artigo/TCC

Cada módulo tem, no próprio docstring, a referência à seção correspondente
do artigo (`artigo.tex`) e ao documento colaborativo do TCC — por exemplo,
`src/agents/contestador.py` explica por que a priorização de contradição
factual vem diretamente do achado dos 43% de "Unclear" no experimento real
da Arabnezhad et al. Se mudar uma escolha arquitetural aqui (ex.: trocar a
Opção B do Juiz por debate multi-rodada), atualize também a seção
correspondente do artigo e do doc.
