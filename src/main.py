from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

class ChurnSentinel:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.model = RandomForestClassifier(n_estimators=100)

    def fit_model(self, target):
        X = self.df.drop(columns=[target])
        y = self.df[target]
        self.model.fit(X, y)
        print("Model trained successfully.")

    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)
        print(classification_report(y_test, preds))

if __name__ == "__main__":
    print("User Churn Sentinel: Random Forest Classification Engine.")
