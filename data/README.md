# Dados

## Decisão: quais datasets vamos usar

Três camadas, cada uma com um papel diferente — não é "escolher um dataset",
é usar os três em momentos diferentes do trabalho:

1. **Sintético (`make_synthetic_corpus` + `alter_ego.py`) — já funciona hoje.**
   Serve só para desenvolvimento/teste de pipeline (ver seção abaixo). Os
   números daqui **não vão para o TCC**, só confirmam que o código roda de
   ponta a ponta antes de gastar tempo com dado real.

2. **Lado "dark web" — The Majestic Garden, via o arquivo público do Gwern
   Branwen (<https://gwern.net/dnm-archive>).** É a mesma fonte usada por
   Ho & Ng (2016), uma das referências do artigo principal, e é
   estruturalmente equivalente ao tipo de fórum que Arabnezhad et al.
   (2020) usaram (fórum de dark web com posts datados por usuário) — mas
   **não é o dataset exato deles**, que não é público (ver nota abaixo).

3. **Lado "internet aberta" — Reddit.** Via API oficial (PRAW, precisa de
   app registrado) para coleta pequena e controlada, ou via dataset já
   coletado (Academic Torrents / Pushshift — checar status de acesso
   atual antes de depender disso, mudou de política algumas vezes).

Com (2) e (3) carregados via `src/data/loaders.py`, usamos a mesma
metodologia de validação "alter-ego" (dividir as mensagens de um usuário
em duas metades, ver `src/data/alter_ego.py`) que Arabnezhad et al.
aplicaram no próprio experimento sintético deles (Seção 4.2 do artigo) —
e, adicionalmente, tentamos pares reais cross-plataforma (um mesmo
usuário que decidiu postar nos dois fóruns, quando identificável por
autodeclaração no texto) como teste mais realista.

## Por que não dá pra usar o dataset exato do artigo principal

O dataset exato que Arabnezhad et al. (2020) usaram (pares de contas
Dream Market / The Majestic Garden coletados por eles em 2017-2019, e o
Reddit correspondente) **não foi disponibilizado publicamente** pelos
autores — foi coletado por eles mesmos, e não há pedido de acesso
documentado no artigo. Isso não bloqueia o trabalho: a metodologia de
validação deles (split "alter-ego") não depende do dataset específico,
só de *um* corpus qualquer com texto e timestamps suficientes por
usuário — é exatamente isso que o item (1) acima já prova, e os itens
(2)/(3) replicam com dado real.

## Como plugar um corpus real

1. Baixe/organize os dados brutos em `data/raw/` (ignorado pelo git —
   ver `.gitignore`; dados de dark web não devem ir pro repositório, por
   tamanho e por sensibilidade).
2. Implemente (ou adapte) um loader em `src/data/loaders.py` — já tem um
   esqueleto `load_reddit_csv` e um stub `load_forum_dump` prontos pra
   apontar pro formato real do arquivo que você baixar (ex.: dump do
   The Majestic Garden do arquivo do Gwern, ou export via PRAW do
   Reddit).
3. Troque a chamada a `make_synthetic_corpus(...)` por `load_reddit_csv(...)`
   (ou o loader que você escrever) em `src/evaluation/run_experiment.py`
   — o resto do pipeline (features, baseline, agentes, métricas) não
   muda nada.

## Dados sintéticos (já funciona hoje)

Enquanto o acesso aos dados reais não está resolvido,
`src/data/loaders.py::make_synthetic_corpus` gera um corpus falso
(mas estruturalmente parecido: usuários com vocabulário e horário de
posting distintos) que já deixa rodar toda a pipeline de ponta a ponta —
é isso que `scripts/run_demo.py` e `python -m src.evaluation.run_experiment`
usam por padrão. Os números que saem disso **não significam nada** para o
TCC — servem só para confirmar que o código funciona antes de gastar
tempo configurando dados reais.
