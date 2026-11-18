"""
05_regression.py — Modélise le log de la dépense par équivalent-adulte (niveau de vie)
en fonction des caractéristiques du ménage : éducation, emploi, démographie, région, milieu.

Utilise une régression linéaire pondérée (poids EHCVM) pour rester représentative,
et OLS statsmodels pour disposer des coefficients, p-values et intervalles de confiance
interprétables (angle d'expert : quelles variables expliquent le niveau de vie, dans quel sens,
avec quelle significativité).

Sortie : reports/regression_summary.txt, reports/figures/regression_coefficients.png
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

BASE = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(BASE, "data", "processed", "menage_clusters.csv")
REPORT_DIR = os.path.join(BASE, "reports")
FIG_DIR = os.path.join(REPORT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

DEP_CANDIDATES = ["dtot", "depense_totale", "depense_eq_adulte", "welfare"]
WEIGHT_CANDIDATES = ["hhweight", "poids_menage", "weight"]
EXPL_CANDIDATES = [
    "taille_menage", "chef_age", "chef_sexe", "chef_niveau_instruction",
    "chef_statut_emploi", "chef_secteur_activite", "milieu", "region",
    "eau_potable", "electricite", "nb_pieces",
]


def run():
    df = pd.read_csv(DATA_PATH)
    dep_col = next((c for c in DEP_CANDIDATES if c in df.columns), None)
    weight_col = next((w for w in WEIGHT_CANDIDATES if w in df.columns), None)
    if dep_col is None:
        raise ValueError("Variable de dépense/bien-être introuvable (vérifier ehcvm_welfare_CIV2018).")

    expl = [c for c in EXPL_CANDIDATES if c in df.columns]
    df = df.dropna(subset=[dep_col])
    df = df[df[dep_col] > 0]
    y = np.log(df[dep_col])

    X = df[expl].copy()
    for c in X.select_dtypes(include="object").columns:
        X[c] = X[c].astype("category").cat.codes
    X = X.fillna(X.median(numeric_only=True))
    X = sm.add_constant(X)

    weights = df[weight_col] if weight_col else np.ones(len(df))

    model = sm.WLS(y, X, weights=weights).fit()
    print(model.summary())

    with open(os.path.join(REPORT_DIR, "regression_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Variable dépendante : log({dep_col}) — niveau de vie (dépense par équivalent-adulte)\n\n")
        f.write(str(model.summary()))

    coefs = model.params.drop("const").sort_values()
    plt.figure(figsize=(7, 5))
    coefs.plot(kind="barh", color=["tab:red" if v < 0 else "tab:green" for v in coefs])
    plt.title("Effets estimés sur le log(dépense par équivalent-adulte)")
    plt.xlabel("Coefficient (WLS)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "regression_coefficients.png"), dpi=150)
    plt.close()

    print(f"\nR² ajusté : {model.rsquared_adj:.3f}")
    return model


if __name__ == "__main__":
    run()
