import os
import joblib

import numpy as np
import pandas as pd
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def main() :
    X_test = pd.read_csv(os.path.expanduser("data/X_test.csv"))
    y_test = pd.read_csv(os.path.expanduser("data/y_test.csv")).values.flatten()
    # espace de 0 à 1 en 100 points
    x = np.linspace(0, 1, 500)
    # On récupère les probabilités positives
    model = joblib.load(os.path.expanduser("data/model.pkl"))
    y_prob = model.predict_proba(X_test)[:, 1]
    precision = []
    recall = []
    for threshold in x:
        y_pred = (y_prob >= threshold).astype(int)
        true_pos = ((y_pred == 1) & (y_test == 1)).sum()
        false_pos = ((y_pred == 1) & (y_test == 0)).sum()
        false_neg = ((y_pred == 0) & (y_test == 1)).sum()
        if true_pos + false_pos != 0 :
            recall.append(true_pos / (true_pos + false_neg))
            precision.append(true_pos / (true_pos + false_pos))
    plt.figure(figsize=(16,11))
    plt.plot(recall, precision)
    plt.title("PR Curve", fontsize=16)
    plt.xlabel("Recall", fontsize=13)
    plt.ylabel("Precision", fontsize=13)
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    plt.savefig("chart/pr_curve.png")
    plt.show()

if __name__ == "__main__" : 
    main()