# Purchase Propensity Prediction

Uma empresa de e-commerce deseja prever a propensão de compra dos usuários com
base em seu comportamento de navegação.

## Dataset utilizado

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
- **Configurações**: Centralizar todas configurações em `configs/config.yaml`.

## Requisitos e instalação

Utilize Python **3.11 ou superior** e Poetry **2.x**. Com ambos disponíveis,
execute na raiz do projeto:

```bash
poetry install
```

## DVC — Versionamento e pipeline de dados

O **DVC (Data Version Control)** é utilizado para orquestrar o pipeline de dados e 
versionar os artefatos gerados em cada etapa.

O pipeline é composto pelas seguintes etapas:

1. **Download:** obtém o dataset original da fonte configurada.
2. **Preprocessamento:** normaliza os nomes das colunas, valida o esquema, normaliza 
valores categóricos e booleanos e remove linhas totalmente vazias e duplicatas exatas.
3. **Preparação dos dados:** cria features derivadas, realiza a separação estratificada 
entre treino e teste e aplica as transformações de features, incluindo imputação e codificação one-hot. 

Para reproduzir o pipeline:

```bash
poetry run dvc repro
```

### Armazenamento dos artefatos

Para este projeto acadêmico, foi escolhido um **remote DVC local**, evitando a necessidade de 
configurar uma infraestrutura de armazenamento em nuvem. 

O remote local pode ser configurado com:

```bash
dvc remote add -d local ../dvc-storage
```

Esse comando define `../dvc-storage` como o armazenamento padrão dos artefatos do DVC. Execute-o uma 
única vez na configuração inicial do projeto; o remote fica registrado na configuração do DVC.

## Como executar

### 1. Reproduzir o pipeline de dados

Na raiz do projeto, execute:

```bash
poetry run dvc repro
```

O DVC executará as etapas necessárias conforme as dependências e os arquivos de 
saída declarados em `dvc.yaml`.

### 2. Executar a aplicação

Após preparar os dados, execute:

```bash
poetry run python main.py
```

O `main.py` consome os arquivos preparados pelo DVC e registra suas dimensões no log.
Ele não executa novamente o download, o preprocessamento, a engenharia de features 
ou a separação entre treino e teste.

## Testes

Para executar a suíte de testes:

```bash
poetry run pytest
```

Os testes utilizam dados sintéticos e arquivos temporários, evitando dependência do 
dataset real ou de serviços externos durante a execução dos testes unitários.

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
