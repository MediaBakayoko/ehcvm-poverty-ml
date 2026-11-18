"""
02_pca.py — ACP sur les conditions de vie des ménages (variables confirmées EHCVM 2018-2019).
Sortie : data/processed/pca_scores.csv, data/processed/pca_loadings.csv,
         reports/figures/pca_variance.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE = os.path.join(os.path.dirname(__file__), "..")
IN_PATH = os.path.join(BASE, "data", "processed", "menage_dataset.csv")
OUT_DIR = os.path.join(BASE, "data", "processed")
FIG_DIR = os.path.join(BASE, "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

LIVING_VARS = [
    "electricite", "eau_potable", "assainissement_ameliore", "possede_tv",
    "possede_refrigerateur", "possede_voiture", "possede_ordinateur",
    "mur", "toit", "sol", "hhsize",
]


def run_pca(n_components=5):
    df = pd.read_csv(IN_PATH)
    X = df[LIVING_VARS].apply(pd.to_numeric, errors="coerce").fillna(0)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(Xs)

    explained = pca.explained_variance_ratio_
    print("Variance expliquée :", np.round(explained, 3), "cumulée :", np.round(np.cumsum(explained), 3))

    plt.figure(figsize=(6, 4))
    plt.bar(range(1, n_components + 1), explained * 100)
    plt.plot(range(1, n_components + 1), np.cumsum(explained) * 100, color="red", marker="o")
    plt.xlabel("Composante"); plt.ylabel("% variance expliquée")
    plt.title("ACP — Variance expliquée (conditions de vie, EHCVM 2018-2019)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "pca_variance.png"), dpi=150)
    plt.close()

    loadings = pd.DataFrame(pca.components_.T, index=LIVING_VARS,
                             columns=[f"PC{i+1}" for i in range(n_components)])
    loadings.to_csv(os.path.join(OUT_DIR, "pca_loadings.csv"))
    print("Variables les plus discriminantes sur PC1 :")
    print(loadings["PC1"].abs().sort_values(ascending=False))

    scores_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(n_components)])
    scores_df = pd.concat([df[["hhid", "region", "milieu"]].reset_index(drop=True), scores_df], axis=1)
    scores_df.to_csv(os.path.join(OUT_DIR, "pca_scores.csv"), index=False)
    return scores_df, loadings, pca


if __name__ == "__main__":
    run_pca()
