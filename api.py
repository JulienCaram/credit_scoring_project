from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import pickle
import shap

app = FastAPI()

# Charger le modèle et les données
model = pickle.load(open("./best_model.pickle", "rb"))
df = pd.read_csv('./test_preprocess.csv')

# Initialiser l'explainer SHAP
explainer = shap.TreeExplainer(model)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """
    <html>
        <head>
            <title>Prédiction de scoring crédit</title>
        </head>
        <body>
            <h2>Entrer l'ID du client pour prédiction</h2>
            <form action="/predict/" method="post">
                <input type="number" name="id_client" />
                <input type="submit" />
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/predict/")
async def predict(id_client: int = Form(...)):
    explanation = None  # Initialiser explanation pour tous les cas
    if id_client not in df['SK_ID_CURR'].unique():
        return {"prediction_text": "Ce client n'est pas répertorié", "Score": None, "Explanation": explanation}
    
    X = df[df['SK_ID_CURR'] == id_client].drop(['SK_ID_CURR'], axis=1)
    probability_default_payment = model.predict_proba(X)[:, 1][0]
    seuil = 0.475
    
    if probability_default_payment >= seuil:
        prediction = "Prêt NON Accordé, risque de défaut élevé."
        
        # Calculer les valeurs SHAP pour l'instance
        shap_values = explainer.shap_values(X)
        # Utiliser [1] pour la classe positive si le modèle est binaire; ajustez selon votre modèle
        index = 1 if model.classes_.shape[0] == 2 else np.argmax(model.predict_proba(X), axis=1)[0]
        shap_values = shap_values[index]
        
        # Identifier la caractéristique la plus impactante
        max_shap_index = np.argmax(np.abs(shap_values), axis=1)[0]
        feature_name = X.columns[max_shap_index]
        feature_value = X.iloc[0, max_shap_index]
        explanation = f"Caractéristique la plus influente: {feature_name} (valeur: {feature_value:.2f})"
    else:
        prediction = "Prêt Accordé"

    return {
        "prediction_text": prediction,
        "Score": float(probability_default_payment),
        "Explanation": explanation
    }

# Pour exécuter API :
# cd Desktop/Python/projet_7
# uvicorn api:app --reload

#to run : http://127.0.0.1:8000
#         http://127.0.0.1:8000/docs