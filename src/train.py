from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib, os

ds = load_dataset("ag_news")
train_texts = ds["train"]["text"]
train_labels = ds["train"]["label"]
test_texts = ds["test"]["text"]
test_labels = ds["test"]["label"]

vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

model = LogisticRegression(max_iter=200)
model.fit(X_train, train_labels)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(test_labels, pred))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.joblib")
joblib.dump(vectorizer, "models/vectorizer.joblib")
#---------------------------------------------------------
#ky program ngarkon datasetin ag_news, pergaditet tekstet duke i shendruar ne vektor me TF-IDF, trajnon nje model Logistic Regression per klasifikimin e teksteve ne kater kategori dhe ruan modelin dhe vectorizer-in e trajnuar ne folderin 'models'.
#Logistic Regression për klasifikim të lajmeve
#mbi test set-in, dhe ruan si modelin ashtu edhe vectorizer-in
#në folderin "models" për t’u përdorur më vonë në predikim.
