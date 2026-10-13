# Figuras

Pasta para imagens usadas no artigo (`artigo.tex`) e na documentação do
repositório: diagrama de arquitetura (Proposer → Contestador → Juiz),
gráficos de resultado (baseline vs. adversarial, taxa de resolução de
"Unclear"), prints do dashboard de experimento, etc.

Convenção de nome: `<secao-ou-tema>-<descricao-curta>.png` (ex.:
`arquitetura-pipeline.png`, `resultados-taxa-unclear.png`), para ficar
fácil referenciar no LaTeX/Markdown sem ambiguidade.

Gráficos gerados por `src/evaluation/run_experiment.py` ou por notebooks
de análise devem ser salvos aqui (não em `data/`, que é só para os
corpora de entrada).
