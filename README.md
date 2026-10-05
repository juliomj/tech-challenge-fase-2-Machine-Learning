# Purchase Propensity Prediction

Uma empresa de e-commerce deseja prever a propensão de compra dos usuários com
base em seu comportamento de navegação.

## Estrutura do projeto

```text
Tech Challenge - Fase 2/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   │   ├── .gitkeep
│   │   └── online_shoppers_intention.csv
│   └── processed/
│       └── .gitkeep
├── models/
│   └── .gitkeep
├── notebooks/
│   └── .gitkeep
├── src/
│   └── purchase_propensity/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   └── preprocessing.py
│       ├── features/
│       │   ├── __init__.py
│       │   └── build_features.py
│       └── utils/
│           ├── __init__.py
│           └── logger.py
├── tests/
│   ├── __init__.py
│   ├── test_loader.py
│   ├── test_preprocessing.py
│   └── test_build_features.py
├── .dockerignore
├── .env.example
├── .gitignore
├── pyproject.toml
├── poetry.lock
├── README.md
└── main.py
```

Os arquivos `.gitkeep` preservam apenas os diretórios inicialmente vazios quando
o projeto for adicionado ao Git. Datasets e modelos locais são ignorados; esta
etapa não implementa versionamento de dados.

## Responsabilidades dos arquivos

| Arquivo | Responsabilidade |
| --- | --- |
| `configs/config.yaml` | Centralizar nome, seed `42`, caminhos e coluna target. |
| `src/purchase_propensity/data/loader.py` | Ler CSV e informar arquivo ausente ou dataset vazio. |
| `src/purchase_propensity/data/preprocessing.py` | Normalizar nomes, remover duplicatas e linhas totalmente ausentes. |
| `src/purchase_propensity/features/build_features.py` | Separar features e target, validando os nomes das colunas. |
| `src/purchase_propensity/utils/logger.py` | Fornecer logging de console com loggers nomeados. |
| `__init__.py` | Identificar e documentar os pacotes Python, sem efeitos colaterais. |
| `main.py` | Demonstrar a composição dos módulos, sem salvar dados ou treinar modelos. |
| `tests/test_loader.py` | Validar leitura de CSV e erros de entrada. |
| `tests/test_preprocessing.py` | Validar as operações independentes de limpeza. |
| `tests/test_build_features.py` | Validar a separação do target em um arquivo dedicado. |
| `pyproject.toml` | Configurar instalação com Poetry, dependências, pytest e Ruff. |
| `poetry.lock` | Fixar as versões resolvidas das dependências para a equipe. |
| `.dockerignore` | Excluir arquivos locais de um futuro contexto de build. |
| `.env.example` | Reservar o modelo de variáveis de ambiente, sem credenciais. |
| `.gitignore` | Excluir ambientes, caches, dados e artefatos locais do Git. |
| `README.md` | Documentar escopo, decisões e comandos da Etapa 1. |
| `.gitkeep` | Manter os diretórios reservados, sem lógica de aplicação. |

`data/processed/`, `models/` e `notebooks/` são apenas espaços reservados. Nenhuma
funcionalidade de persistência, treinamento ou experimentação foi implementada.

A Etapa 1 lê a configuração do YAML e não utiliza variáveis de ambiente.
O `.env.example` registra essa condição para a equipe. O `.dockerignore` define
apenas exclusões de arquivos; a criação do Dockerfile pertence às próximas etapas.

## Boas práticas utilizadas

- **Clean Code e responsabilidade única:** funções pequenas, com leitura,
  limpeza, separação e logging em módulos próprios; `main.py` apenas os compõe.
- **Naming conventions:** funções e variáveis em `snake_case`; constantes em
  `UPPER_CASE`; nomes descritivos em inglês conforme o padrão Python/PEP 8.
- **Type hints:** entradas e retornos tipados nas funções principais, sem `Any`.
- **Docstrings Google Style:** contratos, retornos e erros documentados.
- **Organização modular:** pacote instalável no layout `src/`, sem alterações
  manuais de `sys.path`.
- **Preservação da entrada:** funções de limpeza e separação devolvem cópias e
  preservam o índice, evitando alterações no DataFrame do chamador.
- **Validações claras:** arquivos vazios ou ausentes, target inexistente e nomes
  vazios ou duplicados após normalização geram erros explícitos.
- **Ruff:** verifica estilo, imports, problemas comuns, type hints e docstrings;
  o padrão de docstrings não é exigido nas funções de teste.
- **Testes unitários:** pytest usa arquivos temporários e dados sintéticos
  pequenos, sem depender de um dataset real.

Não há imputação, escalonamento, encoding ou engenharia estatística de features.
A seed está centralizada na configuração para uso futuro; não é aplicada a
treinamento nesta etapa.

## Requisitos e instalação

Utilize Python **3.11 ou superior** e Poetry **2.x**. Com ambos disponíveis,
execute na raiz do projeto:

```bash
poetry install
```

O Poetry instala o pacote em `src/`, as dependências principais (`pandas`,
`numpy`, `scikit-learn`, `pyyaml`) e as de desenvolvimento (`pytest`, `ruff`).
Scikit-learn e NumPy estão declarados para a continuidade do projeto, mas não
há modelo implementado.

Se houver mais de um Python instalado, selecione o interpretador antes:

```bash
poetry env use /caminho/para/python
poetry install
```

No Windows, utilize o caminho do `python.exe`. A seleção só é necessária quando
o Poetry não encontra automaticamente um Python compatível.

## Configuração e dados de entrada

Edite `configs/config.yaml` para ajustar os caminhos, o nome do CSV e o target.
O padrão utiliza `data/raw/online_shoppers_intention.csv` com o target `revenue`.

### Dataset utilizado

O **Online Shoppers Purchasing Intention Dataset** contém 12.330 sessões de
navegação, 17 features e a coluna target `Revenue`, que indica se a sessão
terminou em compra. Entre as features estão páginas visitadas, duração da
navegação, taxas de saída e tipo de visitante. O arquivo CSV tem aproximadamente
1 MB.

- Fonte original e download: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).
- Cópia no [Kaggle](https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset).
- Licença informada pela UCI: **Creative Commons Attribution 4.0 (CC BY 4.0)**.
- Referência: Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention
  Dataset*. UCI Machine Learning Repository.
  [DOI: 10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q).

O CSV foi obtido diretamente da UCI e está disponível localmente em `data/raw/`.
Ele continua ignorado pelo Git. Em outra máquina ou após clonar o projeto,
acesse a página da UCI, clique em **Download**, extraia
`online_shoppers_intention.csv` do ZIP e coloque-o em `data/raw/`.

Nesta etapa, o dataset é utilizado apenas para demonstrar carregamento, limpeza
básica e separação entre features e target.

### Formato de entrada

O CSV deve ter cabeçalho, separador vírgula e pelo menos uma linha de dados. Os
tipos e valores ausentes seguem a leitura padrão do pandas. Nomes como
` Page Views ` e `Revenue` tornam-se `page_views` e `revenue`; configure o
target com o nome **já normalizado**. Rótulos que ficarem vazios ou colidirem
após a normalização serão rejeitados.

Para experimentar com um arquivo menor, salve este **exemplo sintético**
como `data/raw/example.csv`:

```csv
Page Views,Time On Site,Revenue
3,120,1
1,30,0
3,120,1
,,
```

Execute `poetry run python main.py --dataset data/raw/example.csv`. Esse exemplo
resulta em duas linhas, duas features e o target `revenue`.

## Como executar

Com o CSV da UCI em `data/raw/`:

```bash
poetry run python main.py
```

Na validação com o CSV da UCI, o fluxo leu **12.330 linhas**, removeu **125
duplicatas** e retornou **12.205 linhas**, **17 features** e o target `revenue`.
Nenhuma linha foi removida por estar totalmente ausente.

Também é possível indicar outro arquivo sem editar a configuração:

```bash
poetry run python main.py --dataset "caminho/para/dataset.csv"
```

O outro arquivo deve conter o target `revenue` após a normalização; para usar
outro nome de target, ajuste `model.target_column` na configuração.

Os caminhos do YAML são relativos à raiz do projeto; o argumento `--dataset`
é relativo ao diretório de execução ou pode ser absoluto.

O fluxo carrega o CSV, normaliza os nomes das colunas, remove linhas totalmente
ausentes, remove duplicatas e separa features/target. O logging apresenta as
dimensões resultantes. A remoção de linhas vazias preserva linhas parcialmente
preenchidas; strings vazias e espaços não são considerados ausentes pela função
de limpeza, embora o leitor de CSV possa converter campos vazios em `NaN`.

O comando gera um erro claro se o arquivo estiver ausente ou vazio, ou se o
target configurado não existir. Nenhum arquivo processado ou modelo é gerado.

## Testes

```bash
poetry run pytest
```

Casos cobertos: CSV válido, arquivo inexistente, arquivo vazio, somente espaços,
apenas cabeçalho; normalização de nomes e rótulos numéricos; colisões e nomes
vazios; remoção de duplicatas; remoção apenas de linhas totalmente ausentes;
preservação do índice e da entrada; separação correta do target, target ausente
e colunas duplicadas.

## Verificação do código

```bash
poetry run ruff check .
```

Para verificar também a formatação:

```bash
poetry run ruff format --check .
```

Referências de configuração: [Poetry](https://python-poetry.org/docs/pyproject/)
e [Ruff](https://docs.astral.sh/ruff/configuration/).
