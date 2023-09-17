import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score, precision_score

def prepare_dataset():
    # La fonction expanduser permet de convertir le répertoire ~ en chemin absolu
    data = pd.read_csv(os.path.expanduser("data/sample.csv"))
    # On récupère les événements dont la colonne user_session n'est pas nulle
    events_per_session = data.loc[~data["user_session"].isna(), :].copy()

    # On crée une colonne purchase qui vaut 0 ou 1 en fonction de la colonne event_type
    events_per_session['purchased'] = np.where(events_per_session['event_type'] == "purchase", 1, 0)
    # On agrège par session et par produit pour savoir si ce dernier a été acheté dans la session
    events_per_session['purchased'] = events_per_session \
        .groupby(["user_session", "product_id"])['purchased'] \
        .transform("max")

    # On détermine combien de fois l'utilisateur a vu de produits dans la session
    events_per_session['num_views_session'] = np.where(events_per_session['event_type'] == "view", 1, 0)
    events_per_session['num_views_session'] = events_per_session.groupby(["user_session"])['num_views_session'].transform("sum")

    # On détermine combien de fois l'utilisateur a vu un produit en particulier dans la session
    events_per_session['num_views_product'] = np.where(events_per_session['event_type'] == "view", 1, 0)
    events_per_session['num_views_product'] = events_per_session.groupby(["user_session", "product_id"])['num_views_product'].transform("sum")

    events_per_session['category'] = events_per_session['category_code'].str.split(".",expand=True)[0].astype('category')
    events_per_session['sub_category'] = events_per_session['category_code'].str.split(".",expand=True)[1].astype('category')
    events_per_session["event_time"] = pd.to_datetime(events_per_session["event_time"], utc=True)
    events_per_session["hour"] = events_per_session["event_time"].dt.hour
    events_per_session["minute"] = events_per_session["event_time"].dt.minute
    events_per_session["weekday"] = events_per_session["event_time"].dt.dayofweek
    # On calcule la date minimum et maximum de chaque session
    sessions_duration = events_per_session \
        .groupby("user_session").agg(
            {"event_time": [np.min, np.max]}
        ) \
        .reset_index()

    sessions_duration["amax"] = pd.to_datetime(sessions_duration["event_time"]["amax"])
    sessions_duration["amin"] = pd.to_datetime(sessions_duration["event_time"]["amin"])
    del sessions_duration["event_time"]
    # On aplati au niveau 0 les colonnes
    sessions_duration.columns = sessions_duration.columns.get_level_values(0)
    # On calcule la durée totale, en secondes, de chaque TimeDelta
    sessions_duration["duration"] = (sessions_duration["amax"] - sessions_duration["amin"]).dt.seconds
    dataset = events_per_session \
    .sort_values("event_time") \
    .drop_duplicates(["event_type", "product_id", "user_id", "user_session"]) \
    .loc[events_per_session["event_type"].isin(["cart", "purchase"])] \
    .merge(
        sessions_duration[["user_session", "duration"]],
        how="left",
        on="user_session"
    )
    count_prev_sessions = dataset[["user_id", "user_session", "event_time"]] \
        .sort_values("event_time") \
        .groupby(["user_id", "user_session"]) \
        .first() \
        .reset_index()

    # cumcount permet de faire un comptage cumulé des lignes : 0, 1, 2, 3, ...
    count_prev_sessions["num_prev_sessions"] = count_prev_sessions \
        .sort_values("event_time") \
        .groupby("user_id") \
        .cumcount()
    dataset = dataset.merge(
        count_prev_sessions[["user_session", "num_prev_sessions"]],
        how="left",
        on="user_session"
    )
    view_prev_session = dataset[["user_id", "user_session", "event_time", "product_id"]]
    view_prev_session = view_prev_session \
        .sort_values("event_time") \
        .groupby(["user_id", "user_session", "product_id"]) \
        .first() \
        .reset_index()

    view_prev_session["num_prev_product_views"] = view_prev_session \
        .sort_values("event_time") \
        .groupby(["user_id", "product_id"]) \
        .cumcount()

    dataset = dataset.merge(
        view_prev_session[["user_session", "product_id", "num_prev_product_views"]],
        how="left",
        on=["user_session", "product_id"]
    )
    dataset = dataset \
    .sort_values("event_time") \
    .drop_duplicates(["user_session", "product_id", "purchased"]) \
    .drop(["event_time", "event_type", "category_code", "category_id"], axis=1)

    dataset.to_csv(os.path.expanduser("data/dataset.csv"), index=False)
    return(dataset)

def prepare_feature(dataset) :
    features = dataset.drop({"user_id", "user_session"}, axis=1).copy()
    encoder = LabelEncoder()
    for label in ["category", "sub_category", "brand"] :
        features[label] = features[label].fillna("unknown")
        features[label] = features[label].astype(str)
        features[label] = encoder.fit_transform(features[label])
    features['weekday'] = features['weekday'].astype(int)
    X = features.drop(['purchased'], axis=1)
    y = features['purchased']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=40
    )
    print(X_train.head())
    X.to_csv(os.path.expanduser("data/X.csv"), index=False)
    y.to_csv(os.path.expanduser("data/y.csv"), index=False)

    X_train.to_csv(os.path.expanduser("data/X_train.csv"), index=False)
    X_test.to_csv(os.path.expanduser("data/X_test.csv"), index=False)
    y_train.to_csv(os.path.expanduser("data/y_train.csv"), index=False)
    y_test.to_csv(os.path.expanduser("data/y_test.csv"), index=False)
    return(X_train, X_test, y_train, y_test, X, y)

def training(X_train, X_test, y_train, y_test, X, y) :
    # On répète 3 fois un 5-Fold
    y_train = y_train.values.ravel()
    rep_kfold = RepeatedKFold(n_splits=4, n_repeats=3)

    # Hyper-paramètres des modèles
    hyp_params = {
        "num_leaves": 60,
        "min_child_samples": 10,
        "max_depth": 12,
        "n_estimators": 100,
        "learning_rate": 0.1
    }
    scores_train = []
    scores_test = []
    n_iter = 1
    for train_I, test_I in rep_kfold.split(X):
        print("Itération {} du k-Fold".format(n_iter))
        # On récupère les indices des sous-échantillons
        X_fold_train = X.iloc[train_I, :]
        y_fold_train = y.iloc[train_I]
        X_fold_test = X.iloc[test_I, :]
        y_fold_test = y.iloc[test_I]
        
        # On entraîne un LightGBM avec les paramètres par défaut
        model = LGBMClassifier(**hyp_params, objective="binary", verbose=-1)
        model.fit(X_fold_train, y_fold_train.values.ravel())

        # On calcule le score du modèle sur le test
        scores_train.append(
            f1_score(y_fold_train, model.predict(X_fold_train))
        )
        scores_test.append(
            f1_score(y_fold_test, model.predict(X_fold_test))
        )
        n_iter += 1
    print(scores_test, scores_train)
    print("Score Train médian : {:2.1f}%".format(np.mean(scores_train) * 100))
    print("Score Test médian : {:2.1f}%".format(np.mean(scores_test) * 100))
    model = LGBMClassifier(**hyp_params, objective="binary", verbose=-1)
    model.fit(X_train, y_train)
    print("Score Train : {:2.1f}%".format(f1_score(y_train, model.predict(X_train)) * 100))
    print("Score Test : {:2.1f}%".format(f1_score(y_test, model.predict(X_test)) * 100))
    print("Precision Test : {:2.1f}%".format(precision_score(y_test, model.predict(X_test)) * 100))
    print("Recall Test : {:2.1f}%".format(recall_score(y_test, model.predict(X_test)) * 100))

def main() :
    dataset = None
    X_train, X_test, y_train, y_test = None, None, None, None
    X, y = None, None
    if not os.path.exists(os.path.expanduser("data/dataset.csv")) :
        dataset = prepare_dataset()
    else :
        dataset = pd.read_csv(os.path.expanduser("data/dataset.csv"))
    if not os.path.exists(os.path.expanduser('data/X_train.csv')) :
        X_train, X_test, y_train, y_test = prepare_feature(dataset)
    else :
        X_train = pd.read_csv("data/X_train.csv")
        X_test = pd.read_csv("data/X_test.csv")
        y_train = pd.read_csv("data/y_train.csv")
        y_test = pd.read_csv("data/y_test.csv")
        X = pd.read_csv("data/X.csv")
        y = pd.read_csv("data/y.csv")
    training(X_train, X_test, y_train, y_test, X, y)
    
if __name__ == "__main__" :
    main()
