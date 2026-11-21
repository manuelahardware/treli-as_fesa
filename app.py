
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from truss import resolver_trelica, ErroTrelica


st.set_page_config(
    page_title="Análise de Treliças",
    page_icon="🏗️",
    layout="wide"
)

st.title("🏗️ Análise de Treliças")
st.caption("Análise estrutural pelo Método da Rigidez")

st.divider()

# Propriedades do material
st.subheader("1. Propriedades do material")

col1, col2 = st.columns(2)

with col1:
    E = st.number_input(
        "Módulo de elasticidade (Pa)",
        min_value=1.0,
        value=200e9,
        format="%.2e"
    )

with col2:
    A = st.number_input(
        "Área da seção transversal (m²)",
        min_value=0.000001,
        value=0.001,
        format="%.6f"
    )

st.divider()

# Coordenadas dos nós
st.subheader("2. Coordenadas dos nós")
st.caption("Cada linha representa um nó. As coordenadas são dadas em metros.")

nos_df = st.data_editor(
    pd.DataFrame({
        "X (m)": [0.0, 4.0, 2.0],
        "Y (m)": [0.0, 0.0, 3.0]
    }),
    num_rows="dynamic",
    use_container_width=True,
    key="nos"
)

# Barras
st.subheader("3. Conexões das barras")
st.caption("Informe os números dos nós conectados. A numeração começa em 1.")

barras_df = st.data_editor(
    pd.DataFrame({
        "Nó inicial": [1, 1, 2],
        "Nó final": [2, 3, 3]
    }),
    num_rows="dynamic",
    use_container_width=True,
    key="barras"
)

# Apoios
st.subheader("4. Apoios")

apoios_df = st.data_editor(
    pd.DataFrame({
        "Nó": [1, 2],
        "Tipo": ["pino", "rolete_y"]
    }),
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Tipo": st.column_config.SelectboxColumn(
            "Tipo de apoio",
            options=["pino", "rolete_x", "rolete_y"],
            required=True
        )
    },
    key="apoios"
)

# Cargas
st.subheader("5. Forças aplicadas")
st.caption("Valores em Newtons. Forças negativas indicam sentido contrário ao eixo positivo.")

cargas_df = st.data_editor(
    pd.DataFrame({
        "Nó": [3],
        "Fx (N)": [0.0],
        "Fy (N)": [-10000.0]
    }),
    num_rows="dynamic",
    use_container_width=True,
    key="cargas"
)

st.divider()

# Cálculo
if st.button("CALCULAR TRELIÇA", type="primary", use_container_width=True):

    try:
        # Preparação dos nós
        nos_validos = nos_df.dropna()

        nos = nos_validos[["X (m)", "Y (m)"]].astype(float).values.tolist()

        if len(nos) < 2:
            raise ErroTrelica("Cadastre pelo menos dois nós.")

        # Preparação das barras
        barras_validas = barras_df.dropna()

        barras = [
            (int(row["Nó inicial"]) - 1,
             int(row["Nó final"]) - 1)
            for _, row in barras_validas.iterrows()
        ]

        # Preparação dos apoios
        apoios_validos = apoios_df.dropna()

        apoios = {
            int(row["Nó"]) - 1: str(row["Tipo"])
            for _, row in apoios_validos.iterrows()
        }

        # Preparação das cargas
        cargas_validas = cargas_df.dropna()

        cargas = {}

        for _, row in cargas_validas.iterrows():
            no = int(row["Nó"]) - 1

            fx = float(row["Fx (N)"])
            fy = float(row["Fy (N)"])

            if no in cargas:
                cargas[no] = (
                    cargas[no][0] + fx,
                    cargas[no][1] + fy
                )
            else:
                cargas[no] = (fx, fy)

        # Execução do método da rigidez
        resultado = resolver_trelica(
            nos,
            barras,
            apoios,
            cargas,
            E,
            A
        )

        st.success("Treliça calculada com sucesso!")

        # Resultados numéricos
        st.header("6. Resultados")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Nós", len(nos))

        with col2:
            st.metric("Barras", len(barras))

        with col3:
            st.metric("Graus de liberdade", 2 * len(nos))

        # Deslocamentos
        st.subheader("Deslocamentos dos nós")

        deslocamentos = resultado["deslocamentos"]

        tabela_deslocamentos = pd.DataFrame({
            "Nó": range(1, len(nos) + 1),
            "Deslocamento X (m)": deslocamentos[0::2],
            "Deslocamento Y (m)": deslocamentos[1::2]
        })

        st.dataframe(
            tabela_deslocamentos,
            use_container_width=True
        )

        # Reações
        st.subheader("Reações dos apoios")

        tabela_reacoes = pd.DataFrame(
            resultado["reacoes"],
            columns=["Nó", "Direção", "Reação (N)"]
        )

        tabela_reacoes["Nó"] += 1

        st.dataframe(
            tabela_reacoes,
            use_container_width=True
        )

        # Esforços nas barras
        st.subheader("Esforços internos das barras")

        tabela_esforcos = pd.DataFrame({
            "Barra": range(1, len(barras) + 1),
            "Esforço (N)": resultado["esforcos"],
            "Tensão (Pa)": resultado["tensoes"],
            "Classificação": resultado["tipos"]
        })

        st.dataframe(
            tabela_esforcos,
            use_container_width=True
        )

        # Gráfico da treliça
        st.header("7. Visualização da estrutura")

        deslocamentos_xy = resultado["deslocamentos"].reshape(-1, 2)

        max_deslocamento = np.max(
            np.linalg.norm(deslocamentos_xy, axis=1)
        )

        dimensao = max(
            np.ptp(np.array(nos)[:, 0]),
            np.ptp(np.array(nos)[:, 1]),
            1.0
        )

        escala = (
            0.15 * dimensao / max_deslocamento
            if max_deslocamento > 0
            else 1.0
        )

        nos_originais = np.array(nos)
        nos_deformados = nos_originais + deslocamentos_xy * escala

        fig = go.Figure()

        # Estrutura original
        for i, j in barras:
            fig.add_trace(go.Scatter(
                x=[nos_originais[i, 0], nos_originais[j, 0]],
                y=[nos_originais[i, 1], nos_originais[j, 1]],
                mode="lines",
                line=dict(color="gray", width=2, dash="dash"),
                name="Estrutura original",
                showlegend=False
            ))

        # Estrutura deformada
        for k, (i, j) in enumerate(barras):

            tipo = resultado["tipos"][k]

            if tipo == "Tração":
                cor = "red"
            elif tipo == "Compressão":
                cor = "blue"
            else:
                cor = "green"

            fig.add_trace(go.Scatter(
                x=[nos_deformados[i, 0], nos_deformados[j, 0]],
                y=[nos_deformados[i, 1], nos_deformados[j, 1]],
                mode="lines",
                line=dict(color=cor, width=4),
                name=tipo,
                legendgroup=tipo
            ))

        # Nós
        fig.add_trace(go.Scatter(
            x=nos_originais[:, 0],
            y=nos_originais[:, 1],
            mode="markers+text",
            text=[f"Nó {i+1}" for i in range(len(nos))],
            textposition="top center",
            marker=dict(size=10, color="black"),
            name="Nós originais"
        ))

        fig.update_layout(
            title=f"Treliça deformada (escala visual: {escala:.2e}x)",
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            template="plotly_white",
            height=550,
            legend_title="Classificação das barras"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "A linha cinza tracejada representa a estrutura original. "
            "A deformação está ampliada para facilitar a visualização. "
            "Vermelho: tração | Azul: compressão | Verde: esforço nulo."
        )

        # Matriz de rigidez
        with st.expander("Visualizar matriz de rigidez global"):
            st.dataframe(
                pd.DataFrame(resultado["matriz_rigidez"]),
                use_container_width=True
            )

    except ErroTrelica as erro:
        st.error(f"Erro no cálculo: {erro}")

    except Exception as erro:
        st.error(f"Verifique os dados informados: {erro}")
