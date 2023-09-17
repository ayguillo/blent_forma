import os
import numpy as np
import pandas as pd
from hyperopt import hp, tpe, fmin #Attention, hyperopt cherche a MINIMISER sa fonction. Si on optimise par le score, il faut prendre l'inverse
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score, precision_score
import joblib
import warnings
warnings.filterwarnings("ignore")

# définition des specs du model avec les hypers paramètres possibles


def optimize_hyp(instance, training_set, search_space, metric, model_specs, evals=10):
    # Fonction que l'on souhaite minimiser (inverse de tau)
    def objective(params):
        X = training_set[0]
        y = training_set[1]
        for param in set(list(model_specs["override_schemas"].keys())).intersection(set(params.keys())):
            cast_instance = model_specs['override_schemas'][param]
            params[param] = cast_instance(params[param])
            
        # On répète 3 fois un 5-Fold
        rep_kfold = RepeatedKFold(n_splits=4, n_repeats=1)
        scores_test = []
        for train_I, test_I in rep_kfold.split(X):
            X_fold_train = X.iloc[train_I, :]
            y_fold_train = y.iloc[train_I]
            X_fold_test = X.iloc[test_I, :]
            y_fold_test = y.iloc[test_I]

            # On entraîne un LightGBM avec les paramètres par défaut
            model = instance(**params, objective="binary", verbose=-1)
            model.fit(X_fold_train, y_fold_train)

            # On calcule le score du modèle sur le test
            scores_test.append(
                metric(y_fold_test, model.predict(X_fold_test))
            )

        return np.mean(scores_test)

    return fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=evals)


def main() :
    MODEL_SPECS = {
        "name": "LightGBM",
        "class": LGBMClassifier,
        "max_evals": 20,
        "params": {
            "learning_rate": hp.uniform("learning_rate", 0.001, 1),
            "num_iterations": hp.quniform("num_iterations", 100, 1000, 20),
            "max_depth": hp.quniform("max_depth", 4, 12, 6),
            "num_leaves": hp.quniform("num_leaves", 8, 128, 10),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.3, 1),
            "subsample": hp.uniform("subsample", 0.5, 1),
            "min_child_samples": hp.quniform("min_child_samples", 1, 20, 10),
            "reg_alpha": hp.choice("reg_alpha", [0, 1e-1, 1, 2, 5, 10]),
            "reg_lambda": hp.choice("reg_lambda", [0, 1e-1, 1, 2, 5, 10]),
        },
        "override_schemas": {
            "num_leaves": int, "min_child_samples": int, "max_depth": int, "num_iterations": int
        }
    }
    X_train = pd.read_csv(os.path.expanduser("data/X_train.csv"))
    X_test = pd.read_csv(os.path.expanduser("data/X_test.csv"))
    y_train = pd.read_csv(os.path.expanduser("data/y_train.csv"))
    y_test = pd.read_csv(os.path.expanduser("data/y_test.csv")).values.flatten()
    X = pd.read_csv(os.path.expanduser("data/X.csv"))
    y = pd.read_csv(os.path.expanduser("data/y.csv"))

    optimum_params = optimize_hyp(
        MODEL_SPECS['class'],
        training_set=(X_train, y_train),
        search_space=MODEL_SPECS["params"],
        metric=lambda x, y: -f1_score(x, y), # Problème de minimisation = inverse du score
        evals=MODEL_SPECS["max_evals"],
        model_specs=MODEL_SPECS
        )
    for param in MODEL_SPECS['override_schemas']:
        cast_instance = MODEL_SPECS['override_schemas'][param]
        optimum_params[param] = cast_instance(optimum_params[param])
    model = LGBMClassifier(**optimum_params)
    model.fit(X_train, y_train)
    print("F1 Score : {:2.1f}%".format(f1_score(y_test, model.predict(X_test)) * 100))
    print("Precision : {:2.1f}%".format(precision_score(y_test, model.predict(X_test)) * 100))
    print("Recall : {:2.1f}%".format(recall_score(y_test, model.predict(X_test)) * 100))
    joblib.dump(model, os.path.expanduser("data/model.pkl"))
if __name__ == "__main__" :
    main()