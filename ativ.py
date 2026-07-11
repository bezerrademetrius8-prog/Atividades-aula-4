import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# 1. Configuração Inicial da Página do Streamlit
st.set_page_config(page_title="Preditor de Notas", page_icon="🎓")
st.title("🎓 Preditor de Notas com Rede Neural")
st.write("Descubra se o seu esforço vai render aquele notão ou um puxão de orelha!")

# 2. Criação automatizada do arquivo CSV (Garante projeto 100% novo e independente)
CSV_FILE = "dados_estudo.csv"
if not os.path.exists(CSV_FILE):
    dados = {
        "Horas_Estudo": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "Nota": [2.0, 3.0, 4.2, 4.8, 5.2, 5.8, 6.5, 7.0, 7.8, 8.5, 9.2, 9.8, 10.0, 10.0]
    }
    pd.DataFrame(dados).to_csv(CSV_FILE, index=False)

# 3. Leitura e Preparação dos Dados com Pandas e Numpy
df = pd.read_csv(CSV_FILE)
X = df[['Horas_Estudo']].values
y = df['Nota'].values

# 4. Normalização dos Dados (Essencial para Redes Neurais)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# 5. Treinamento da Rede Neural (Scikit-Learn)
@st.cache_resource
def treinar_rede_neural(X_s, y_s):
    # MLPRegressor configurado para aprender o padrão das notas
    mlp = MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=2500, random_state=42)
    mlp.fit(X_s, y_s)
    return mlp

modelo = treinar_rede_neural(X_scaled, y_scaled)
st.success("🤖 Rede Neural calibrada e pronta para o cálculo!")

# 6. Interface Gráfica com o Usuário (Streamlit)
st.markdown("### 🕒 Quantas horas você vai estudar por dia?")
horas = st.slider("Arraste para definir as horas:", min_value=0.0, max_value=12.0, value=4.0, step=0.5)

# 7. Botão de Ação e Lógica de Análise
if st.button("🔮 Calcular Minha Nota Futura"):
    # Convertendo a entrada para o formato correto e aplicando escala
    horas_array = np.array([[horas]])
    horas_scaled = scaler_X.transform(horas_array)
   
    # Predição e retorno para a escala original de notas
    pred_scaled = modelo.predict(horas_scaled)
    nota_calculada = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
   
    # Limitando a nota estritamente entre 0 e 10
    nota_calculada = max(0.0, min(10.0, float(nota_calculada)))
   
    # Exibição do resultado numérico arredondado
    nota_final = round(nota_calculada, 1)
    st.subheader(f"Nota Calculada pela IA: {nota_final}")
   
    # 8. Sistema de Classificação por Níveis
    if nota_final < 5.0:
        st.error(
            "🚨 **Classificação: Nota Ruim!**\n\n"
            "Ih, essa quantidade de horas não tá legal. "
            "Hora de largar o celular e focar nos livros!"
        )
    elif 5.0 <= nota_final <= 6.0:
        st.warning(
            "⚠️ **Classificação: Nota OK.**\n\n"
            "Na média! Dá para passar de ano, mas você tem potencial "
            "para brilhar muito mais se apertar o passo."
        )
    elif 6.0 < nota_final < 10.0:
        st.info(
            "😎 **Classificação: Nota Boa!**\n\n"
            "Muito bem! O esforço está valendo a pena. "
            "Continua assim que o sucesso no boletim é garantido!"
        )
    elif nota_final == 10.0:
        st.success(
            "🔥 **Classificação: Nota Ótima!**\n\n"
            "PERFEITO! Você atingiu o nível mestre dos estudos. "
            "Sua nota calculada foi um 10 absoluto!"
        )

# 9. Rodapé informativo do app
st.caption("Desenvolvido com Python, Scikit-Learn, Pandas, NumPy e Streamlit.")
