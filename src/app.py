from fastapi import FastAPI
from pydantic import BaseModel
import joblib, os, json

app = FastAPI()

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
model = joblib.load(os.path.join(MODELS_DIR, "model.joblib"))
vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.joblib"))

AG_NEWS_ID2LABEL = {0: "Bota", 1: "Sporti", 2: "Biznes", 3: "Shkence"}
id2label = AG_NEWS_ID2LABEL

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    X = vectorizer.transform([input.text])
    pred_id = int(model.predict(X)[0])
    return {"prediction": id2label.get(pred_id, str(pred_id))}

#-----------------------------------------------
# Ky aplikacion perdor fastapi per te ndertuar nje api per klasifikim teksti
# ngarkon modelin e trajnuar Logistic Regression dhe TF-IDF vectorizerin nga folderi "models
# perdor nje skeme pydantic per te validuar inputin e tekstit"
# dhe ofron nje endpoint "/predict" qe pranon nje tekst dhe kthen kategorine e parashikuar.