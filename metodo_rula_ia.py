import streamlit as st
import google.generativeai as genai
import mediapipe as mp
import cv2
import numpy as np
import os
import math

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.title("📸 Análise Ergonômica com RULA - Perfil Único")

uploaded_file = st.file_uploader("Envie uma imagem de perfil (jpg ou png)", type=["jpg", "jpeg", "png"])

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Funções de pontuação RULA
def rula_tronco(angle): return 1 if angle <= 5 else 2 if angle <= 20 else 3 if angle <= 60 else 4
def rula_pescoco(angle): return 1 if angle <= 10 else 2 if angle <= 20 else 3
def rula_antebraco(angle): return 1 if 60 < angle < 100 else 2
def rula_braco(angle): return 1 if angle <= 20 else 2 if angle <= 45 else 3 if angle <= 90 else 4

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption='Imagem enviada', use_column_width=True)

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            st.success("✅ Postura detectada com sucesso!")
            landmarks = results.pose_landmarks.landmark
            height, width, _ = image.shape
            annotated = image.copy()

            def pt(idx): return int(landmarks[idx].x * width), int(landmarks[idx].y * height)

            right_vis = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility
            left_vis = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility

            if right_vis > left_vis:
                side = "Direito"
                s = mp_pose.PoseLandmark.RIGHT_SHOULDER
                e = mp_pose.PoseLandmark.RIGHT_ELBOW
                w = mp_pose.PoseLandmark.RIGHT_WRIST
                h = mp_pose.PoseLandmark.RIGHT_HIP
                ear = mp_pose.PoseLandmark.RIGHT_EAR
            else:
                side = "Esquerdo"
                s = mp_pose.PoseLandmark.LEFT_SHOULDER
                e = mp_pose.PoseLandmark.LEFT_ELBOW
                w = mp_pose.PoseLandmark.LEFT_WRIST
                h = mp_pose.PoseLandmark.LEFT_HIP
                ear = mp_pose.PoseLandmark.LEFT_EAR

            ps, pe, pw, ph, pear = pt(s), pt(e), pt(w), pt(h), pt(ear)

            # Cálculo de ângulos
            ang_tronco = int(calculate_angle(ps, ph, [ph[0], ph[1] - 100]))
            ang_pescoco = int(calculate_angle(ps, pear, ph))
            ang_antebraco = int(calculate_angle(ps, pe, pw))
            ang_braco = int(calculate_angle(ph, ps, pe))

            # RULA scores
            r_tronco = rula_tronco(ang_tronco)
            r_pescoco = rula_pescoco(ang_pescoco)
            r_antebraco = rula_antebraco(ang_antebraco)
            r_braco = rula_braco(ang_braco)

            # Desenhar conexões
            cv2.line(annotated, ph, ps, (255, 0, 0), 3)  # tronco - azul
            cv2.line(annotated, ps, pe, (0, 0, 255), 3)  # braco - vermelho
            cv2.line(annotated, pe, pw, (0, 255, 255), 3)  # antebraco - amarelo
            cv2.line(annotated, ps, pear, (0, 255, 0), 3)  # pescoco - verde

            # Pontos
            for p in [ph, ps, pe, pw, pear]:
                cv2.circle(annotated, p, 6, (0, 255, 0), -1)

            # Texto superior
            text_lines = [
                f"Tronco {side}: {ang_tronco} (RULA {r_tronco})",
                f"Pescoco {side}: {ang_pescoco} (RULA {r_pescoco})",
                f"Antebraco {side}: {ang_antebraco} (RULA {r_antebraco})",
                f"Braco {side}: {ang_braco} (RULA {r_braco})"
            ]
            for i, txt in enumerate(text_lines):
                cv2.putText(annotated, txt, (20, 40 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Mostrar imagem final
            st.image(annotated, caption="Imagem com análise RULA", use_column_width=True)

            _, img_encoded = cv2.imencode(".png", annotated)
            st.download_button(
                label="📥 Baixar imagem com pontos e RULA",
                data=img_encoded.tobytes(),
                file_name="imagem_rula.png",
                mime="image/png"
            )

            # Gemini explicação
            prompt = f"""
Você é um especialista em ergonomia. Baseado na imagem analisada, os ângulos e pontuações RULA foram:

- Tronco: {ang_tronco}° (RULA {r_tronco})
- Pescoço: {ang_pescoco}° (RULA {r_pescoco})
- Antebraço: {ang_antebraco}° (RULA {r_antebraco})
- Braço: {ang_braco}° (RULA {r_braco})

Explique:
1. Estime os ângulos dos segmentos (braço, antebraço, tronco, pescoço).
2. Use as regras da tabela RULA para estimar a pontuação final.
3. Diga qual segmento está em pior posição.
4. Dê ao menos uma sugestão de ajuste postural ou ambiental.
"""
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            st.subheader("📊 Análise da IA:")
            st.markdown(response.text)
        else:
            st.error("❌ Não foi possível detectar a postura.")
