# Text Classification Starter

Ky projekt implementon nje model klasifikimi te teksteve duke perdorur dataset-in AG_NEWS.

Kam ndare ne 4 kategori te ndryshme dhe bazuar ne tekstin qe apo informacionin qe ne japim ai klasifikohet ne ato 4 kategori te cilat jane:

-Sports
-Business
-Sci/Tech
-Worl

Modeli eshte trajnuar me scikit-learn dhe TF-IDF vectorizer, dhe sherbehet permes nje API me FastAPI.

- Teknologjite e perdoruare: Python, FastAPI, scikit-learn, TF-IDF vectorizer, joblib, datasets, pandas, numpy, matplotlib, uvicorn.

Si te ekzekutohet projekti:

1- Krijo nje enviroment virtual
py -m venv .venv
.venv\Scripts\activate

2- Instalo modulet
pip install -r requirements.txt

3- Trajno modelin
py src\train.py

4- Testo nje parashikim per modelin
py src\predict.py "Real Madrid secure dramatic win in Champions League"


Ekzekutimi i API:

Nise serverin : uvicorn src.app:app --reload

Shembull kerkese: curl -X POST "http://127.0.0.1:8000/predict" `
     -H "Content-Type: application/json" `
     -d "{\"text\":\"NASA launches new Mars rover\"}"

dhe pastaj do te marresh pergjigjen tek prediction
