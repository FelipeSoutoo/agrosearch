# AgroSearch

O **AgroSearch** é um protótipo de motor de busca textual. A aplicação
realiza recuperação de informações em uma pequena base de documentos
relacionados à agricultura.

O projeto integra conceitos de **Processamento de Linguagem Natural
(PLN)** e **Recuperação de Informação**, incluindo pré-processamento
textual, índice invertido, ranqueamento TF-IDF e Similaridade de
Cosseno.

## Funcionalidades

-   Tokenização dos documentos e da consulta;
-   Normalização do texto para letras minúsculas e remoção de acentos;
-   Remoção opcional de stopwords;
-   Aplicação opcional de stemming;
-   Construção de vocabulário;
-   Construção de índice direto e índice invertido;
-   Busca textual;
-   Cálculo manual de TF, DF, IDF e TF-IDF;
-   Ranking dos documentos por TF-IDF acumulado;
-   Similaridade de Cosseno utilizando vetores TF-IDF;
-   Interface interativa desenvolvida com Streamlit.

## Tecnologias utilizadas

-   Python
-   Streamlit
-   Pandas

O índice invertido, o TF-IDF e a Similaridade de Cosseno foram
implementados diretamente no código, sem o uso de bibliotecas de alto
nível como `TfidfVectorizer` do scikit-learn.

## Estrutura do projeto

``` text
AgroSearch/
├── app.py
├── requirements.txt
├── Relatorio_AgroSearch.pdf
├── README.md
└── .gitignore
```

A pasta `.venv` utilizada no desenvolvimento é local e não deve ser
enviada ao repositório.

## Como executar

### 1. Clonar o repositório

``` bash
git clone URL_DO_REPOSITORIO
cd AgroSearch
```

### 2. Criar o ambiente virtual

No Windows:

``` bash
python -m venv .venv
```

### 3. Ativar o ambiente virtual

Prompt de Comando (CMD):

``` bash
.venv\Scripts\activate
```

PowerShell:

``` powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Instalar as dependências

``` bash
python -m pip install -r requirements.txt
```

### 5. Executar a aplicação

``` bash
python -m streamlit run app.py
```

Após a execução, o Streamlit disponibilizará a aplicação no navegador.

## Funcionamento

### Fase 1 --- Pré-processamento

Os documentos passam por tokenização e normalização. O usuário também
pode ativar ou desativar a remoção de stopwords e o stemming pela
interface.

### Fase 2 --- Índice Invertido

A aplicação constrói uma estrutura que relaciona cada termo processado
aos documentos nos quais ele aparece:

``` text
Termo -> [Documentos]
```

Também é exibido o índice direto:

``` text
Documento -> [Termos]
```

### Fase 3 --- TF-IDF

A consulta recebe o mesmo pré-processamento aplicado aos documentos.
Para cada termo são calculados:

``` text
TF = frequência do termo / quantidade de termos do documento
IDF = log(N / DF)
TF-IDF = TF * IDF
```

Os valores são acumulados por documento e utilizados para gerar o
ranking de relevância.

### Bônus --- Similaridade de Cosseno

A aplicação também representa a consulta e os documentos como vetores
TF-IDF e calcula a Similaridade de Cosseno:

``` text
cos(A, B) = (A · B) / (||A|| * ||B||)
```

Esse resultado é exibido em um ranking separado do TF-IDF acumulado.

## Exemplos de consultas

Algumas consultas que podem ser utilizadas para testar a aplicação:

``` text
irrigação
irrigação soja
lagartas soja
nitrogênio milho
cultivo orgânico
computador
```

## Relatório

O arquivo `Relatorio_AgroSearch.pdf` contém uma descrição resumida da
implementação, dos testes realizados e da divisão das atividades.

## Integrantes

-   Felipe Souto Maior Mendes
-   Arthur Xavier Cavalcante
-   Julliane Di Paula Oliveira Xavier
