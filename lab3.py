import numpy as np
import pandas as pd


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class CustomLogisticRegression:
    def __init__(self, lr=0.01, num_epochs=1000):
        self.lr = lr
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.num_epochs):
            linear_model = np.dot(X,self.weights) + self.bias
            y_predicted = sigmoid(linear_model)

            dw = (1/ n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/ n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        y_predicted_cls = [1 if i>= threshold else 0 for i in self.predict_proba(X)]
        return np.array(y_predicted_cls)


class TabularDataset:
    """Handles loading tsv files, splitting features/targets, and standardization."""
    def __init__(self, train_path, test_path, target_col):
        train_df = pd.read_csv(train_path, sep="\t")
        test_df = pd.read_csv(test_path, sep="\t")

        self.X_train = train_df.drop(columns=[target_col]).values
        self.y_train = train_df[target_col].values

        self.X_test = test_df.drop(columns=[target_col]).values
        self.y_test = test_df[target_col].values

        self.mean = self.X_train.mean(axis=0)
        self.std = self.X_train.std(axis=0)
        self.std[self.std == 0] = 1e-8

        self.X_train_scaled = (self.X_train - self.mean) / self.std
        self.X_test_scaled = (self.X_test - self.mean) / self.std

def evaluate_predictions(y_true, y_pred):
    """computes and prints standard binary classification metrics"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2* (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print(f"  Accuracy:  {accuracy * 100:.2f}%")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print("  Confusion Matrix:")
    print(f"    [ TN: {tn:<4} FP: {fp:<4} ]")
    print(f"    [ FN: {fn:<4} TP: {tp:<4} ]\n")


datasets = [
    {
        "name": "Heart disease",
        "data": TabularDataset("heart_train.tsv", "heart_test.tsv", "heart_disease"),
    },
    {
        "name":"Education",
        "data": TabularDataset("education_train.tsv", "education_test.tsv", "grade"),
    },
]

for item in datasets:
    name = item["name"]
    dataset = item["data"]

    model = CustomLogisticRegression(lr = 0.1, num_epochs=500)
    model.fit(dataset.X_train_scaled, dataset.y_train)

    y_preds = model.predict(dataset.X_test_scaled)

    print(f"=== {name} Dataset Evaluation ===")
    evaluate_predictions(dataset.y_test, y_preds)
