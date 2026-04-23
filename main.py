# ============================================================
# API REST - DETECCIÓN DE DISLEXIA
# Modelo Ensemble de Fusión por Mediana (RF + LR + SVM)
# Desplegado en Render.com
# ============================================================

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import os

app = FastAPI(
    title="API Detección Dislexia - Ensemble Mediana",
    description="Aplicativo Móvil con Modelo Ensemble de Fusión por Mediana para Detección de Dislexia",
    version="2.0"
)

# ============================================================
# CARGA DE LOS 3 MODELOS DEL ENSEMBLE
# ============================================================
print("🔄 Cargando modelo ensemble...")

obj = joblib.load('ensemble_model.pkl')

if isinstance(obj, tuple) and len(obj) == 3:
    rf_model = obj[0]   # Random Forest
    lr_model = obj[1]   # Regresión Logística
    svm_model = obj[2]  # SVM
    print("✅ 3 modelos cargados correctamente:")
    print(f"   - Random Forest: {type(rf_model).__name__}")
    print(f"   - Regresión Logística: {type(lr_model).__name__}")
    print(f"   - SVM: {type(svm_model).__name__}")
else:
    raise ValueError("El archivo .pkl no contiene una tupla con 3 modelos")


# ============================================================
# FUSIÓN POR MEDIANA
# ============================================================
def predecir_fusion_mediana(df):
    """
    1. Cada modelo (RF, LR, SVM) calcula probabilidad de clase 1 (Riesgo)
    2. Se calcula la mediana de las 3 probabilidades
    3. Umbral de decisión 0.5
    """
    prob_rf = rf_model.predict_proba(df)[:, 1]
    prob_lr = lr_model.predict_proba(df)[:, 1]
    prob_svm = svm_model.predict_proba(df)[:, 1]

    probabilidades = np.vstack([prob_rf, prob_lr, prob_svm])
    mediana_prob = np.median(probabilidades, axis=0)

    prediccion = (mediana_prob >= 0.5).astype(int)

    return prediccion, mediana_prob, prob_rf, prob_lr, prob_svm


# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/")
def home():
    return {
        "mensaje": "API funcionando correctamente",
        "modelo": "Ensemble Fusión por Mediana (RF + LR + SVM)",
        "version": "2.0",
        "status": "online"
    }


@app.get("/health")
def health():
    """Endpoint para verificar que el servidor está vivo"""
    return {"status": "ok"}


class DatosNino(BaseModel):
    # 7 variables principales del modelo
    edad: float
    tiempo_lectura_seg: float
    errores_lectura: float
    comprension_score_0_10: float
    omisiones_silabas: float
    inversiones_letras: float
    grado_num: float

    # 4 variables de eye-tracking (se guardan pero no van al modelo)
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    fixation_time: float = 0.0
    saccades: float = 0.0


@app.post("/predict")
def predict(data: DatosNino):
    try:
        df = pd.DataFrame([{
            "Edad": data.edad,
            "Tiempo_Lectura_seg": data.tiempo_lectura_seg,
            "Errores_Lectura": data.errores_lectura,
            "Comprension_Score_0_10": data.comprension_score_0_10,
            "Omisiones_Silabas": data.omisiones_silabas,
            "Inversiones_Letras": data.inversiones_letras,
            "Grado_Num": data.grado_num
        }])

        if df.isnull().values.any():
            return {"success": False, "error": "Datos incompletos"}

        pred, mediana, p_rf, p_lr, p_svm = predecir_fusion_mediana(df)

        resultado = "Riesgo de Dislexia" if pred[0] == 1 else "Sin Riesgo"

        return {
            "success": True,
            "resultado": resultado,
            "codigo_modelo": int(pred[0]),
            "probabilidad_mediana": round(float(mediana[0]), 4),
            "probabilidades_individuales": {
                "random_forest": round(float(p_rf[0]), 4),
                "regresion_logistica": round(float(p_lr[0]), 4),
                "svm": round(float(p_svm[0]), 4)
            },
            "tiempo": data.tiempo_lectura_seg,
            "eye_tracking": {
                "gaze_x": data.gaze_x,
                "gaze_y": data.gaze_y,
                "fixation_time": data.fixation_time,
                "saccades": data.saccades
            }
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================
# ARRANQUE
# ============================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
