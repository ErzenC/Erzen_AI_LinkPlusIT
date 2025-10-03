from fastapi import FastAPI #Importimi i FastAPI dhe BaseModel
from pydantic import BaseModel #Importimi pydantic behet per te validuar inputet - tekstet qe duam ti klasifikojme
import joblib #Importimi i joblib behet per te ngarkuar modelin dhe vectorizer-in e trajnuar

app = FastAPI() #Inicializimi i aplikacionit FastAPI

#Ngarkojme modelin e trajnjuar dhe vectorizer-in nga Folderi 'models'
model = joblib.load("models/model.joblib")
vectorizer = joblib.load("models/vectorizer.joblib")

#Krijimi i klasaes schemes per inputin e tekstit
class Item(BaseModel):
    text: str

#Krijimi i endpointit te parashikimit
@app.post("/predict")
def predict(item: Item):
    X = vectorizer.transform([item.text])
    pred = model.predict(X)[0]
    return {"label": int(pred)}
