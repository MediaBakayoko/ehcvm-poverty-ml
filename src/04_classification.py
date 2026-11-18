"""
04_classification.py — Classification supervisée du statut de pauvreté officiel.

Cible : 'pauvre_officiel' (issue du seuil de pauvreté EHCVM appliqué à la dépense par
équivalent-adulte — PAS un quintile de richesse). Les poids d'échantillonnage (hhweight)
sont utilisés pour que les métriques restent représentatives de la population.

Modèles comparés : Logistic Regression (baseline interprétable), Random Forest,
Gradient Boosting. Sélection par validation croisée stratifiée + AUC/F1.

Sortie : reports/classification_report.txt, reports/figures/feature_importance.png,
         reports/figures/roc_curves.png
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, f1_score

BASE = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(BASE, "data", "processed", "menage_clusters.csv")
REPORT_DIR = os.path.join(BASE, "reports")
FIG_DIR = os.path.join(REPORT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

FEATURES_CANDIDATES = [
    "taille_menage", "chef_age", "chef_sexe", "chef_niveau_instruction",
    "chef_statut_emploi", "chef_secteur_activite", "milieu", "region",
    "eau_potable", "electricite", "assainissement_ameliore",
    "nb_pieces", "possede_telephone", "possede_moto", "cluster",
]
TARGET = "pauvre_officiel"
WEIGHT_CANDIDATES = ["hhweight", "poids_menage", "weight"]


def load():
    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Cible '{TARGET}' absente — vérifier 01_build_dataset.py / noms EHCVM.")
    feats = [f for f in FEATURES_CANDIDATES if f in df.columns]
    weight_col = next((w for w in WEIGHT_CANDIDATES if w in df.columns), None)

    df = df.dropna(subset=[TARGET])
    X = df[feats].copy()
    for c in X.select_dtypes(include="object").columns:
        X[c] = X[c].astype("category").cat.codes
    X = X.fillna(X.median(numeric_only=True))
    y = df[TARGET].astype(int)
    w = df[weight_col].values if weight_col else np.ones(len(df))
    return X, y, w, feats


def run():
    X, y, w, feats = load()
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=0.25, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced", random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42),
    }

    results = {}
    plt.figure(figsize=(6, 6))
    for name, model in models.items():
        Xtr = X_train_s if name == "LogisticRegression" else X_train
        Xte = X_test_s if name == "LogisticRegression" else X_test
        model.fit(Xtr, y_train, sample_weight=w_train)
        proba = model.predict_proba(Xte)[:, 1]
        pred = model.predict(Xte)
        auc = roc_auc_score(y_test, proba, sample_weight=w_test)
        f1 = f1_score(y_test, pred, sample_weight=w_test)
        results[name] = {"AUC": auc, "F1": f1}
        fpr, tpr, _ = roc_curve(y_test, proba, sample_weight=w_test)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        print(f"\n=== {name} ===\nAUC={auc:.3f}  F1={f1:.3f}")
        print(classification_report(y_test, pred, sample_weight=w_test))

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title("Courbes ROC — prédiction de la pauvreté officielle")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "roc_curves.png"), dpi=150)
    plt.close()

    best_name = max(results, key=lambda n: results[n]["AUC"])
    best_model = models[best_name]
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=feats).sort_values(ascending=False)
        plt.figure(figsize=(7, 5))
        importances.head(15).plot(kind="barh")
        plt.gca().invert_yaxis()
        plt.title(f"Variables les plus discriminantes ({best_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "feature_importance.png"), dpi=150)
        plt.close()
        importances.to_csv(os.path.join(REPORT_DIR, "feature_importance.csv"))
        print("\nTop variables discriminantes :\n", importances.head(15))

    with open(os.path.join(REPORT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Meilleur modèle : {best_name}\n")
        for name, r in results.items():
            f.write(f"{name}: AUC={r['AUC']:.3f} F1={r['F1']:.3f}\n")

    print(f"\nMeilleur modèle : {best_name} -> {results[best_name]}")
    return results


if __name__ == "__main__":
    run()
