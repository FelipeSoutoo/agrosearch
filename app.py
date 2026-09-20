import streamlit as st
import re
import unicodedata
from collections import defaultdict
import math
import pandas as pd


# ============================================================
# BASE DE DOCUMENTOS
# ============================================================

documentos = {
    "Doc 1": "A soja requer irrigação constante durante o período de floração para garantir a produtividade.",
    "Doc 2": "O controle biológico de lagartas na soja pode ser feito com a vespa Trichogramma.",
    "Doc 3": "A adubação verde com leguminosas melhora o nitrogênio no solo para o milho.",
    "Doc 4": "Lagartas desfolhadoras causam grande prejuízo na cultura da soja e do algodão.",
    "Doc 5": "A irrigação por gotejamento economiza água e é ideal para o cultivo orgânico."
}


# ============================================================
# STOPWORDS
# ============================================================

stopwords = [
    "a", "o", "as", "os",
    "de", "da", "do", "das", "dos",
    "em", "na", "no", "nas", "nos",
    "para", "por", "com",
    "e", "um", "uma"
]


# ============================================================
# FUNÇÕES DE PRÉ-PROCESSAMENTO
# ============================================================

def normalizar(texto):
    texto = texto.lower()

    texto = unicodedata.normalize("NFD", texto)
    texto = texto.encode("ascii", "ignore").decode("utf-8")

    return texto


def stemmer_simples(palavra):
    if palavra.endswith("entos"):
        return palavra[:-4]

    if palavra.endswith("amente"):
        return palavra[:-5]

    if palavra.endswith("ados"):
        return palavra[:-3]

    return palavra


def preprocessar(texto, remover_stopwords=True, aplicar_stemming=True):

    # 1. Tokenização
    tokens = re.findall(r"\b\w+\b", texto)

    # 2. Normalização
    tokens_normalizados = [
        normalizar(token)
        for token in tokens
    ]

    # 3. Remoção de Stopwords
    if remover_stopwords:
        tokens_processados = [
            token
            for token in tokens_normalizados
            if token not in stopwords
        ]
    else:
        tokens_processados = tokens_normalizados

    # 4. Stemming
    if aplicar_stemming:
        tokens_processados = [
            stemmer_simples(token)
            for token in tokens_processados
        ]

    return tokens_processados

def construir_indice_invertido(documentos_processados):

    indice_invertido = defaultdict(list)

    for id_documento, termos in documentos_processados.items():

        for termo in termos:

            if id_documento not in indice_invertido[termo]:
                indice_invertido[termo].append(id_documento)

    return dict(indice_invertido)

def calcular_tfidf(query, documentos_processados,
                   remover_stopwords=True,
                   aplicar_stemming=True):

    # Aplica à consulta o mesmo pré-processamento dos documentos
    termos_query = preprocessar(
        query,
        remover_stopwords=remover_stopwords,
        aplicar_stemming=aplicar_stemming
    )

    numero_documentos = len(documentos_processados)

    resultados = []

    for id_documento, termos_documento in documentos_processados.items():

        tfidf_acumulado = 0

        for termo in termos_query:

            # Frequência do termo no documento
            frequencia = termos_documento.count(termo)

            # TF(t, d)
            if len(termos_documento) > 0:
                tf = frequencia / len(termos_documento)
            else:
                tf = 0

            # DF(t)
            df = sum(
                1
                for termos in documentos_processados.values()
                if termo in termos
            )

            # IDF(t)
            if df > 0:
                idf = math.log(numero_documentos / df)
            else:
                idf = 0

            # TF-IDF
            tfidf = tf * idf

            tfidf_acumulado += tfidf

        resultados.append({
            "Documento": id_documento,
            "TF-IDF Acumulado": round(tfidf_acumulado, 4)
        })

    # Ordena do maior para o menor
    resultados.sort(
        key=lambda resultado: resultado["TF-IDF Acumulado"],
        reverse=True
    )

    return termos_query, resultados

def calcular_similaridade_cosseno(
    query,
    documentos_processados,
    remover_stopwords=True,
    aplicar_stemming=True
):

    # Pré-processa a consulta
    termos_query = preprocessar(
        query,
        remover_stopwords=remover_stopwords,
        aplicar_stemming=aplicar_stemming
    )

    numero_documentos = len(documentos_processados)

    # Cria o vocabulário usando os documentos e a query
    vocabulario = set()

    for termos in documentos_processados.values():
        vocabulario.update(termos)

    vocabulario.update(termos_query)

    vocabulario = sorted(vocabulario)

    # --------------------------------------------------------
    # Calcula o IDF de cada termo
    # --------------------------------------------------------

    idfs = {}

    for termo in vocabulario:

        df = sum(
            1
            for termos in documentos_processados.values()
            if termo in termos
        )

        if df > 0:
            idfs[termo] = math.log(numero_documentos / df)
        else:
            idfs[termo] = 0

    # --------------------------------------------------------
    # Vetor TF-IDF da Query
    # --------------------------------------------------------

    vetor_query = []

    for termo in vocabulario:

        frequencia = termos_query.count(termo)

        if len(termos_query) > 0:
            tf = frequencia / len(termos_query)
        else:
            tf = 0

        vetor_query.append(
            tf * idfs[termo]
        )

    # --------------------------------------------------------
    # Vetores dos documentos + Similaridade
    # --------------------------------------------------------

    resultados = []

    for id_documento, termos_documento in documentos_processados.items():

        vetor_documento = []

        for termo in vocabulario:

            frequencia = termos_documento.count(termo)

            if len(termos_documento) > 0:
                tf = frequencia / len(termos_documento)
            else:
                tf = 0

            vetor_documento.append(
                tf * idfs[termo]
            )

        # Produto escalar
        produto_escalar = sum(
            q * d
            for q, d in zip(vetor_query, vetor_documento)
        )

        # Norma da Query
        norma_query = math.sqrt(
            sum(valor ** 2 for valor in vetor_query)
        )

        # Norma do Documento
        norma_documento = math.sqrt(
            sum(valor ** 2 for valor in vetor_documento)
        )

        # Similaridade de Cosseno
        if norma_query > 0 and norma_documento > 0:
            similaridade = (
                produto_escalar /
                (norma_query * norma_documento)
            )
        else:
            similaridade = 0

        resultados.append({
            "Documento": id_documento,
            "Similaridade": round(similaridade, 4)
        })

    resultados.sort(
        key=lambda resultado: resultado["Similaridade"],
        reverse=True
    )

    return resultados

# ============================================================
# INTERFACE
# ============================================================

st.title("🌱 AgroSearch")
st.subheader("Motor de Busca Inteligente")

st.write(
    "Sistema de recuperação de informações em documentos "
    "sobre agricultura."
)

st.divider()


# ============================================================
# CONFIGURAÇÕES
# ============================================================

st.subheader("⚙️ Configurações do Pré-processamento")

usar_stopwords = st.checkbox(
    "Remover Stopwords",
    value=True
)

usar_stemming = st.checkbox(
    "Aplicar Stemming",
    value=True
)

st.divider()


# ============================================================
# DOCUMENTOS ORIGINAIS
# ============================================================

st.subheader("📚 Base de Documentos")

for id_documento, texto in documentos.items():
    st.write(f"**{id_documento}:** {texto}")

st.divider()


# ============================================================
# FASE 1 - PRÉ-PROCESSAMENTO
# ============================================================

st.header("Fase 1 - Pré-processamento")

documentos_processados = {}

for id_documento, texto in documentos.items():

    tokens = preprocessar(
        texto,
        remover_stopwords=usar_stopwords,
        aplicar_stemming=usar_stemming
    )

    documentos_processados[id_documento] = tokens

    with st.expander(id_documento):
        st.write("**Texto original:**")
        st.write(texto)

        st.write("**Tokens processados:**")
        st.write(tokens)


# ============================================================
# VOCABULÁRIO
# ============================================================

vocabulario = set()

for tokens in documentos_processados.values():
    vocabulario.update(tokens)

vocabulario = sorted(vocabulario)

st.subheader("🔤 Vocabulário")

st.write(vocabulario)

st.write(
    f"**Quantidade de termos no vocabulário:** "
    f"{len(vocabulario)}"
)

# ============================================================
# FASE 2 - ÍNDICE INVERTIDO
# ============================================================

st.divider()

st.header("Fase 2 - Índice Invertido")

indice_invertido = construir_indice_invertido(
    documentos_processados
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Índice Direto")
    st.write("Documento → Termos")
    st.json(documentos_processados)

with col2:
    st.subheader("Índice Invertido")
    st.write("Termo → Documentos")
    st.json(indice_invertido)

# ============================================================
# FASE 3 - BUSCA E RANQUEAMENTO TF-IDF
# ============================================================

st.divider()

st.header("Fase 3 - Busca e Ranqueamento TF-IDF")

query = st.text_input(
    "Digite sua consulta:",
    placeholder="Ex.: irrigação soja"
)

if st.button("Buscar"):

    # --------------------------------------------------------
    # Validação da consulta
    # --------------------------------------------------------

    if not query.strip():
        st.warning("Digite uma consulta para realizar a busca.")

    else:

        # ====================================================
        # RANKING TF-IDF
        # ====================================================

        termos_query, resultados = calcular_tfidf(
            query,
            documentos_processados,
            remover_stopwords=usar_stopwords,
            aplicar_stemming=usar_stemming
        )

        st.subheader("Consulta processada")

        st.write(termos_query)

        # Cria tabela com os resultados
        df_resultados = pd.DataFrame(resultados)

        st.subheader("Ranking dos Documentos")

        st.dataframe(
            df_resultados,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # Documento vencedor pelo TF-IDF
        # ----------------------------------------------------

        if (
            resultados
            and resultados[0]["TF-IDF Acumulado"] > 0
        ):

            vencedor = resultados[0]["Documento"]

            st.success(
                f"🏆 Documento mais relevante: {vencedor}"
            )

            st.write(
                f"**Conteúdo:** {documentos[vencedor]}"
            )

        else:

            st.info(
                "Nenhum documento relevante foi encontrado "
                "para a consulta."
            )

        # ====================================================
        # DESAFIO BÔNUS - SIMILARIDADE DE COSSENO
        # ====================================================

        st.divider()

        st.subheader("Similaridade de Cosseno")

        resultados_cosseno = calcular_similaridade_cosseno(
            query,
            documentos_processados,
            remover_stopwords=usar_stopwords,
            aplicar_stemming=usar_stemming
        )

        # Cria tabela com os resultados
        df_cosseno = pd.DataFrame(resultados_cosseno)

        st.dataframe(
            df_cosseno,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # Documento vencedor pela Similaridade de Cosseno
        # ----------------------------------------------------

        if (
            resultados_cosseno
            and resultados_cosseno[0]["Similaridade"] > 0
        ):

            vencedor_cosseno = (
                resultados_cosseno[0]["Documento"]
            )

            st.success(
                f"Documento mais similar: "
                f"{vencedor_cosseno}"
            )

        else:

            st.info(
                "Nenhum documento apresentou similaridade "
                "com a consulta."
            )