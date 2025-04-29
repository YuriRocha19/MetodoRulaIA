import streamlit as st
import google.generativeai as genai
import mediapipe as mp
import cv2
import numpy as np
import os

# Configurar a API Key do Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.title("Assistente de Ergonomia - Análise RULA")

uploaded_file = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Ler imagem
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption='Imagem enviada', use_column_width=True)

    # Detectar poses usando MediaPipe
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            # Aqui você pode fazer análise de ângulos, etc.
            st.success("Postura detectada com sucesso!")
            st.write("Calculando RULA Score...")

            # Exemplo básico: enviar a descrição dos pontos para o Gemini
            landmarks_text = str(results.pose_landmarks)

            model = genai.GenerativeModel('gemini-2.0-flash')

            prompt = f"""
Você é um assistente de ergonomia. 
Baseado nos seguintes pontos de referência do corpo extraídos de uma imagem (no formato abaixo), estime um RULA Score e justifique.
Pontos:
{landmarks_text}

Explique qual membro está em pior posição e sugira melhorias ergonômicas.
"""

            response = model.generate_content(prompt)

            st.subheader("Resultado da Análise:")
            st.write(response.text)

        else:
            st.error("Não foi possível detectar a postura. Tente outra imagem.")
