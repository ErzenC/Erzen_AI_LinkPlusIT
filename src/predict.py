import argparse
import json
import os
import sys
import joblib

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

AG_NEWS_ID2LABEL = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

def load_artifacts():
    model_path = os.path.join(MODELS_DIR, "model.joblib")
    vec_path = os.path.join(MODELS_DIR, "vectorizer.joblib")
    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        print("Error: model/vectorizer not found in 'models/'. Run 'py src\\train.py' first.", flush=True)
        sys.exit(1)
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)

    # optional label map
    id2label_path = os.path.join(MODELS_DIR, "label_map.json")
    if os.path.exists(id2label_path):
        with open(id2label_path, "r", encoding="utf-8") as f:
            id2label_loaded = json.load(f)
        id2label = {int(k): v for k, v in id2label_loaded.items()}
    else:
        id2label = AG_NEWS_ID2LABEL

    return model, vectorizer, id2label

def main(): 
    parser = argparse.ArgumentParser(description="Predict category for a given text.")
    parser.add_argument("text", type=str, help="Input text to classify (wrap in quotes)")
    parser.add_argument("--proba", action="store_true", help="Show class probabilities")
    args = parser.parse_args()

    model, vectorizer, id2label = load_artifacts()

    X = vectorizer.transform([args.text])
    pred_id = int(model.predict(X)[0])
    label = id2label.get(pred_id, str(pred_id))

    print(f"Prediction: {label}", flush=True)

    if args.proba and hasattr(model, "predict_proba"):
        import numpy as np
        proba = model.predict_proba(X)[0]
        top = np.argsort(proba)[::-1][:4]
        pretty = [f"{id2label.get(int(i), str(i))}: {proba[i]:.3f}" for i in top]
        print("Top probabilities:", ", ".join(pretty), flush=True)

if __name__ == "__main__":
    main()

 #-----------------------------------------------
 # Ky program merr nje tekst si input nga terminali
 # dhe parashikon kategorine e tij duke perdorur modelin e trajnuar.
 # Shembull i thirrjes nga terminali:
 #   py src\predict.py "The stock market crashed today due to economic uncertainty." --proba
 # -----------------------------------------------   