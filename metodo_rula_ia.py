import streamlit as st
import google.generativeai as genai
import mediapipe as mp
import cv2
import numpy as np
import os
import math

# Configurar a API Key do Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.title("📸 Assistente de Ergonomia - Análise RULA com IA")

uploaded_file = st.file_uploader("Envie uma imagem (foto frontal)", type=["jpg", "jpeg", "png"])

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption='Imagem enviada', use_column_width=True)

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            st.success("✅ Postura detectada com sucesso!")

            annotated_image = image.copy()
            height, width, _ = annotated_image.shape
            landmarks = results.pose_landmarks.landmark

            # Obter coordenadas dos pontos
            def get_point(idx):
                return int(landmarks[idx].x * width), int(landmarks[idx].y * height)

            # Lado dominante (baseado em visibilidade)
            right_score = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility
            left_score = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility

            if right_score > left_score:
                side = "Direito"
                shoulder = mp_pose.PoseLandmark.RIGHT_SHOULDER
                elbow = mp_pose.PoseLandmark.RIGHT_ELBOW
                wrist = mp_pose.PoseLandmark.RIGHT_WRIST
                hip = mp_pose.PoseLandmark.RIGHT_HIP
                ear = mp_pose.PoseLandmark.RIGHT_EAR
            else:
                side = "Esquerdo"
                shoulder = mp_pose.PoseLandmark.LEFT_SHOULDER
                elbow = mp_pose.PoseLandmark.LEFT_ELBOW
                wrist = mp_pose.PoseLandmark.LEFT_WRIST
                hip = mp_pose.PoseLandmark.LEFT_HIP
                ear = mp_pose.PoseLandmark.LEFT_EAR

            # Coordenadas
            p_shoulder = get_point(shoulder)
            p_elbow = get_point(elbow)
            p_wrist = get_point(wrist)
            p_hip = get_point(hip)
            p_ear = get_point(ear)

            # Cálculo de ângulos
            ang_braco = calculate_angle(p_hip, p_shoulder, p_elbow)
            ang_antebraco = calculate_angle(p_shoulder, p_elbow, p_wrist)
            ang_tronco = calculate_angle(p_shoulder, p_hip, [p_hip[0], p_hip[1] - 100])
            ang_pescoco = calculate_angle(p_shoulder, p_ear, p_hip)

            # Desenhar landmarks
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            # Desenhar ângulos
            cv2.putText(annotated_image, f'Braco ({side}): {int(ang_braco)}°', (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated_image, f'Antebraco: {int(ang_antebraco)}°', (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
            cv2.putText(annotated_image, f'Tronco: {int(ang_tronco)}°', (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 0), 2)
            cv2.putText(annotated_image, f'Pescoco: {int(ang_pescoco)}°', (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

            # Exibir imagem final
            st.image(annotated_image, caption="📍 Imagem com ângulos e pontos", use_column_width=True)

            # Download
            _, img_encoded = cv2.imencode(".png", annotated_image)
            st.download_button(
                label="📥 Baixar imagem com análise",
                data=img_encoded.tobytes(),
                file_name="imagem_rula_analisada.png",
                mime="image/png"
            )

            # Análise com Gemini
            landmarks_text = str(results.pose_landmarks)
            prompt = f"""
Você é um assistente de ergonomia. Baseado nos pontos corporais abaixo (MediaPipe), estime um RULA Score e justifique.

Pontos:
{landmarks_text}

1. Estime os ângulos dos segmentos (braço, antebraço, tronco, pescoço).
2. Use as regras da tabela RULA para estimar a pontuação final.
3. Diga qual segmento está em pior posição.
4. Dê ao menos uma sugestão de ajuste postural ou ambiental.
"""
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)

            st.subheader("📊 Análise RULA com IA:")
            st.markdown(response.text)

        else:
            st.error("❌ Não foi possível detectar a postura. Tente outra imagem com melhor qualidade.")
