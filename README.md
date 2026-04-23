# API Detección de Dislexia

API REST basada en Modelo Ensemble de Fusión por Mediana (Random Forest + Regresión Logística + SVM) para la detección temprana de dislexia en estudiantes de educación primaria.

## Tecnologías
- FastAPI
- scikit-learn
- Modelo Ensemble (RF + LR + SVM) con fusión por mediana

## Endpoints
- `GET /` — Estado del servicio
- `GET /health` — Health check
- `POST /predict` — Predicción de riesgo de dislexia

## Variables de entrada (POST /predict)
- edad, tiempo_lectura_seg, errores_lectura
- comprension_score_0_10, omisiones_silabas, inversiones_letras, grado_num
- gaze_x, gaze_y, fixation_time, saccades (eye-tracking)

## Proyecto
Tesis: "Aplicativo Móvil con Modelo Ensemble de Fusión por Mediana para Detección de Dislexia en una Institución Educativa de Piura 2025"
