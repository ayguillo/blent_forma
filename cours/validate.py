# Valeur de shapeley -> shap install

import os
import joblib

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
import seaborn as sns
import shap
import warnings

warnings.filterwarnings("ignore")

sns.set()

def calibration_curve_plt(model, X_test, y_test) :
    prob_pos = model.predict_proba(X_test)[:, 1]
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=20)

    plt.figure(figsize=(16, 10))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", alpha=0.6)
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    plt.ylabel("Fraction of positives")
    plt.xlabel("Predicted probabilites")
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    plt.title("Calibration Curve")
    plt.savefig("chart/calibration_curve.png")
    plt.show()

def density_chart(y_prob, y_test) :
    
    plt.figure(figsize=(16, 10))

    sns.histplot(y_prob[y_test == 0, 1], alpha=0.5)
    plt.axvline(np.median(y_prob[y_test == 0, 1]), 0,1, linestyle="--", label="Median Class 0")
    plt.axvline(np.mean(y_prob[y_test == 0, 1]), 0,1, linestyle="-", label="Mean Class 0")

    sns.histplot(y_prob[y_test == 1, 1], color="darkorange", alpha=0.4)
    plt.axvline(np.median(y_prob[y_test == 1, 1]), 0, 1, linestyle="--", color="darkorange", label="Median Class 1")
    plt.axvline(np.mean(y_prob[y_test == 1, 1]), 0, 1, linestyle="-", color="darkorange", label="Mean Class 1")

    plt.legend()
    plt.xlabel("Predicted probabilites")
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    plt.xlim(-0.05, 1.05)
    plt.title("Density Chart", fontsize=16)
    plt.savefig("chart/density_chart.png")
    plt.show()

def roc_curve_plt(y_test, model, X_test) :

    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])

    plt.figure(figsize=(16, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:2.1f}%)'.format(auc(fpr, tpr) * 100))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    plt.title("ROC Curve", fontsize=16)
    plt.legend(loc="lower right")
    plt.savefig("chart/roc_curve.png")
    plt.show()

def precision_recall_curve_plt(y_test, model, X_test) :
    y_pred = model.predict_proba(X_test)

    plt.figure(figsize=(16,11))
    prec, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:,1], pos_label=1)
    pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot(ax=plt.gca())
    plt.title("PR Curve", fontsize=16)
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    plt.savefig("chart/pr_curve.png")
    plt.show()

def plot_shapley_values(index, X_shap, shap_values, explainer):
    shap_df = pd.DataFrame.from_dict({
        'Variable': X_shap.columns + " (" + X_shap.iloc[0, :].values.astype(str) + ")",
        'Valeur de Shapley': shap_values[index, :]
    })

    # Pour rappel, la prédiction est égale à la somme des valeurs de Shapley + la valeur moyenne
    prob = explainer.expected_value[1] + shap_df['Valeur de Shapley'].sum()
    prob = 1 / (1 + np.exp(-prob))

    plt.figure(figsize=(13,10))
    sns.barplot(
        y='Variable',
        x='Valeur de Shapley',
        data=shap_df.sort_values('Valeur de Shapley', ascending=False)
    )
    plt.title(
        "Probabilité : {:2.2f}%".format(prob * 100),
        fontsize=18
    )
    plt.yticks(fontsize=13)
    plt.savefig("chart/shapley_values.png")
    plt.show()
    
def define_shapeley(model, X_train, X_test) :
    explainer = shap.TreeExplainer(model)
    X_shap = X_test.copy()
    # On récupère les valeurs de Shapley dans la matrice (pour la proba positive)
    shap_values = explainer.shap_values(X_shap)[1]
    plot_shapley_values(1, X_shap, shap_values, explainer)
    shap.summary_plot(shap_values, X_shap, plot_size=0.8)
    shap.dependence_plot("product_id", shap_values, X_shap, interaction_index=None)
    shap.dependence_plot("hour", shap_values, X_shap, interaction_index=None)
    shap.dependence_plot("num_views_session", shap_values, X_shap, interaction_index=None)


def main() :
    model = joblib.load(os.path.expanduser("data/model.pkl"))
    X_train = pd.read_csv(os.path.expanduser("data/X_train.csv"))
    X_test = pd.read_csv(os.path.expanduser("data/X_test.csv"))
    y_train = pd.read_csv(os.path.expanduser("data/y_train.csv")).values.flatten()
    y_test = pd.read_csv(os.path.expanduser("data/y_test.csv")).values.flatten()

    y_prob = model.predict_proba(X_test)
    # density_chart(y_prob, y_test)
    # calibration_curve_plt(model, X_test, y_test)
    # roc_curve_plt(y_test, model, X_test)
    # precision_recall_curve_plt(y_test, model, X_test)
    define_shapeley(model, X_train, X_test)

if __name__ == "__main__" :
   main()