# Sentiment Analysis baseline (not doneatall)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Tiny seed dataset so the pipeline runs immediately (positive=1, negative=0)
samples = [
    ("I loved this movie, fantastic acting and plot.", 1),
    ("Terrible movie, Waste of time.", 0),
    ("An absolute masterpiece, would watch again.", 1),
    ("The script was weak and the pacing was awful.", 0),
    ("Great soundtrack and performances.", 1),
    ("I did not enjoy it, Very boring.", 0),
    ("Brilliant direction, emotional story.", 1),
    ("Bad effects and worse dialogue.", 0),
    ("Surprisingly good and entertaining.", 1),
    ("Predictable and dull.", 0),
    ("A heartfelt film with strong characters.", 1),
    ("Confusing plot and messy editing.", 0),
]
data = pd.DataFrame(samples, columns=["review", "label"])

X_train, X_test, y_train, y_test = train_test_split(
    data["review"], data["label"], test_size=0.25, random_state=42, stratify=data["label"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)
preds = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, preds), 3))
print(classification_report(y_test, preds, digits=3))

# Quick demo predictions
demo_texts = [
    "This was amazing, I loved every minute.",
    "The worst film I have ever seen."
]
demo_preds = model.predict(demo_texts)
for t, p in zip(demo_texts, demo_preds):
    print(f"> '{t}' -> {'positive' if p==1 else 'negative'}")
