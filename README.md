# Purchase Propensity Prediction

Tech Challenge — Fase 2 (11MLET). Pipeline de Machine Learning de ponta a ponta
para prever a propensão de compra de usuários de um e-commerce, com foco em
Engenharia de ML: Clean Code, Poetry, DVC, MLflow e Docker.

**Vídeo de apresentação (STAR):** _[link](https://youtu.be/dCatf8hEa7s)_

## Integrantes

| Integrante | Etapa |
|---|---|
| Julio Matuura | 1 — Fundação do projeto e ambiente |
| Alexandre Borges | 2 — Dados e pipeline DVC |
| Grazielle Nascimento | 3 — Modelagem e avaliação |
| Brendon Bachie | 4 — MLflow, Model Registry e Docker |
| Gabriel Cordeiro | 5 — Consolidação, qualidade e entrega |

## O problema

Uma empresa de e-commerce quer identificar, a partir do comportamento de
navegação, quais sessões têm maior propensão a terminar em compra. É um problema
de **classificação binária**: a sessão gerou receita (`revenue = 1`) ou não.

## Dataset

O **Online Shoppers Purchasing Intention Dataset** contém 12.330 sessões de
navegação, 17 features e a coluna target `Revenue`. Entre as features estão
páginas visitadas por tipo, duração da navegação, taxas de rejeição e de saída,
valor das páginas, mês e tipo de visitante. A classe positiva é minoritária
(cerca de 15% das sessões terminam em compra).

- Fonte: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
  (cópia no [Kaggle](https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset)).
- Licença: **Creative Commons Attribution 4.0 (CC BY 4.0)**.
- Referência: Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention
  Dataset*. UCI Machine Learning Repository.
  [DOI: 10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q).

## Estrutura do projeto

```
.
├── configs/config.yaml        # caminhos, seed, hiperparâmetros e nomes do MLflow
├── data/                      # raw/, processed/, prepared/ (gerados pelo DVC)
├── docs/                      # decisões e entregas por etapa
├── metrics/metrics.json       # métricas de teste versionadas pelo DVC
├── models/                    # modelo treinado (gerado pelo DVC)
├── notebooks/eda.ipynb        # análise exploratória
├── src/purchase_propensity/
│   ├── data/                  # download, loader, schema e pré-processamento
│   ├── features/              # engenharia de features e split treino/teste
│   ├── models/                # train, evaluate, tracking (MLflow) e registry
│   └── utils/                 # configuração e logging
├── tests/                     # testes unitários (pytest)
├── dvc.yaml / dvc.lock        # pipeline e estado versionado
├── Dockerfile / docker-compose.yml
├── pyproject.toml / poetry.lock
├── .env.example
├── main.py                    # confere os dados preparados pelo pipeline
└── predict_sample.py          # envia sessões de teste ao modelo servido
```

## Boas práticas utilizadas

- **Clean Code:** funções curtas (até 20 linhas) e com uma responsabilidade;
  cada estágio do pipeline é um módulo com uma função principal e um `main()`.
- **Naming conventions:** `snake_case` para funções e variáveis, `UPPER_CASE` para
  constantes, nomes descritivos em inglês (PEP 8).
- **Type hints e docstrings** (Google Style) em todas as funções do pacote.
- **Programação funcional:** as transformações recebem um DataFrame e devolvem
  uma cópia, sem alterar a entrada do chamador.
- **Schema único:** grupos de colunas e a separação features/target ficam em
  `data/schema.py`, compartilhados por todos os estágios.
- **Configuração externa:** parâmetros em `configs/config.yaml`; variáveis de
  ambiente em `.env` (só o `.env.example` é versionado).
- **Seed fixa** (`project.random_seed = 42`) no split e no modelo.
- **Ruff** com todas as regras (`select = ["ALL"]`) e **pytest** com dados
  sintéticos, sem depender do dataset real nem de serviços externos.
- **Finais de linha LF** (`.gitattributes`), para que os hashes do `dvc.lock`
  sejam iguais em Windows, Linux e Docker.

## Instalação

Requisitos: Python **3.11+** e Poetry **2.x** (`pipx install poetry`).

```bash
poetry install
```

Grupos de dependências do `pyproject.toml`:

- **main (produção):** pandas, numpy, scikit-learn, MLflow, PyYAML, python-dotenv, joblib.
- **pipeline:** DVC, necessário para executar o pipeline (também instalado no Docker).
- **dev:** pytest, Ruff, ipykernel e matplotlib, usados só em desenvolvimento.

## Configuração

```bash
cp .env.example .env   # opcional
```

| Onde | O quê |
|---|---|
| `.env` | `MLFLOW_TRACKING_URI`. Sem ele, o MLflow usa `sqlite:///mlflow.db` na raiz. Para usar o servidor do Docker a partir do host: `http://localhost:5000` |
| `configs/config.yaml` | caminhos dos dados, seed, `test_size`, hiperparâmetros do Random Forest e nomes do experimento/modelo no MLflow |

## Pipeline (DVC)

```
download_raw → preprocess → build_features → train → evaluate → track → register
```

| Estágio | O que faz | Saída |
|---|---|---|
| `download_raw` | baixa o ZIP da UCI e extrai o CSV | `data/raw/` |
| `preprocess` | normaliza nomes de colunas, valida o schema, normaliza categóricas e booleanas, remove linhas vazias e duplicadas | `data/processed/` |
| `build_features` | cria features derivadas, separa treino/teste (estratificado, 80/20), imputa e aplica one-hot | `data/prepared/train.csv`, `test.csv` |
| `train` | treina o `RandomForestClassifier` com `model.params` e seed fixa | `models/model.joblib` |
| `evaluate` | calcula accuracy, precision, recall, F1 e ROC-AUC no teste | `metrics/metrics.json` |
| `track` | registra parâmetros, métricas, dataset de teste e modelo numa run do MLflow | `metrics/mlflow_run.json` |
| `register` | registra o modelo no Model Registry e o promove a `@champion` se superar o campeão atual | `metrics/registered_model.json` |

Executar o pipeline completo:

```bash
poetry run dvc repro
```

O DVC só reexecuta os estágios cujas dependências (código, dados ou parâmetros)
mudaram. Depois de rodar, `poetry run dvc status` deve responder
`Data and pipelines are up to date`.

Conferir os dados preparados:

```bash
poetry run python main.py
```

### Experimentos com outros parâmetros

Cada experimento gera uma nova run no MLflow e é avaliado contra o campeão:

```bash
poetry run dvc exp run -S configs/config.yaml:model.params.max_depth=20
poetry run dvc exp show
```

### Versionamento dos dados

O `dvc.lock` guarda o hash (md5) de cada arquivo de dados, do modelo e das
métricas. Ele é commitado no Git, então cada commit aponta para uma versão exata
dos dados. Os arquivos em si ficam fora do Git (no cache do DVC).

O remote configurado (`.dvc/config`) é um **remote local**, a pasta
`../dvc-storage` ao lado do repositório. Ele foi escolhido para não depender de
infraestrutura em nuvem. Por isso, `dvc pull` só funciona na máquina que tem essa
pasta. Em uma máquina nova, os dados são recuperados com `dvc repro`: o estágio
`download_raw` baixa o dataset da UCI, e o hash registrado no `dvc.lock` garante
que é a mesma versão usada pelo grupo.

Para enviar os artefatos ao remote local depois de rodar o pipeline:

```bash
poetry run dvc push
```

## MLflow — Tracking e Model Registry

- **Tracking:** cada execução do estágio `track` cria uma run no experimento
  `purchase_propensity` com todos os hiperparâmetros, a seed, o `test_size`, as
  métricas de teste, o dataset de avaliação e o modelo (com assinatura e exemplo
  de entrada).
- **Model Registry:** estratégia *champion/challenger*. A run nova só vira uma
  versão de `purchase_propensity_classifier` e recebe o alias `@champion` se o F1
  (`mlflow.selection_metric`) for maior que o do campeão atual somado a
  `mlflow.min_improvement`. A decisão fica em `metrics/registered_model.json`.

Abrir a interface localmente:

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Acesse http://localhost:5000: **Experiments** mostra as runs e **Models** mostra
o Registry, com a versão marcada como `@champion`.

## Docker

O `Dockerfile` instala os grupos `main` e `pipeline` via Poetry (sem o grupo dev),
roda como usuário não root e executa `dvc repro`. O `docker-compose.yml` usa a
mesma imagem em três serviços:

| Serviço | Função | Porta |
|---|---|---|
| `mlflow` | Tracking server e Model Registry (SQLite + artefatos em volume) | 5000 |
| `pipeline` | Executa o pipeline DVC completo e loga no `mlflow` | - |
| `serve` | Serve o modelo `@champion` via REST (perfil `serve`) | 8000 |

```bash
docker compose build
docker compose up -d mlflow             # UI em http://localhost:5000
docker compose run --rm pipeline        # pipeline completo, do download ao Registry
docker compose --profile serve up -d serve
docker compose --profile serve down     # o volume mlflow-data mantém o histórico
```

O `configs/` é montado no container: alterar `config.yaml` no host e rodar
`docker compose run --rm pipeline` de novo gera uma nova run e reavalia o campeão.

Com o `serve` rodando e os dados preparados localmente, o script abaixo envia 5
sessões do conjunto de teste para a API e compara a previsão com o resultado real:

```bash
poetry run python predict_sample.py
```

A API também aceita chamadas diretas (colunas de `data/prepared/test.csv`, sem
`revenue`):

```bash
curl -X POST http://localhost:8000/invocations -H "Content-Type: application/json" -d '{"dataframe_split": {"columns": ["..."], "data": [[...]]}}'
```

## Testes e qualidade

```bash
poetry run pytest
poetry run ruff check .
poetry run ruff format --check .
```

## Resultados

Após remover duplicatas, restaram 12.205 sessões: 9.764 para treino e 2.441
para teste. Métricas do modelo padrão (`n_estimators=200`, `max_depth=10`,
`min_samples_leaf=2`, `class_weight=balanced`) no conjunto de teste:

| Métrica | Valor |
|---|---|
| Accuracy | 0.866 |
| Precision | 0.548 |
| Recall | 0.840 |
| F1 | 0.663 |
| ROC-AUC | 0.925 |

O `class_weight=balanced` prioriza o recall: o modelo encontra 84% das sessões
que compram, ao custo de mais falsos positivos. Para campanhas de remarketing,
deixar de identificar um comprador costuma custar mais do que abordar alguém que
não compraria.

Experimentos rastreados no MLflow (F1 é a métrica de seleção):

| Run | Parâmetros | F1 | ROC-AUC | Registry |
|---|---|---|---|---|
| base | `max_depth=10`, `n_estimators=200` | 0.663 | 0.925 | v1 → `@champion` |
| exp1 | `max_depth=20`, `n_estimators=300` | **0.679** | 0.926 | v2 → `@champion` |
| exp2 | `max_depth=4`, `n_estimators=300` | 0.613 | 0.900 | não promovido |

O pipeline executado do zero no Docker (Linux, Python 3.12) reproduz exatamente
as métricas da execução local (Windows, Python 3.13).

Detalhes das decisões de cada etapa em [`docs/`](docs/).

## Referências

[Poetry](https://python-poetry.org/docs/pyproject/) ·
[Ruff](https://docs.astral.sh/ruff/configuration/) ·
[DVC](https://dvc.org/doc) ·
[MLflow](https://mlflow.org/docs/latest/)
