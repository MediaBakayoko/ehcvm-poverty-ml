"""
02_pca.py — Analyse en Composantes Principales (ACP) sur les conditions de vie des ménages.

Variables typiques utilisées (adapter selon disponibilité réelle dans menage_dataset.csv) :
- Accès à l'eau potable, à l'électricité, à l'assainissement amélioré
- Matériau du logement (murs, toit, sol)
- Possession d'équipements (radio, TV, réfrigérateur, moto, véhicule, téléphone)
- Nombre de pièces / personnes par pièce (densité d'occupation)
- Éducation du chef de ménage, taille du ménage

Sortie : data/processed/pca_scores.csv, reports/figures/pca_variance.png, pca_biplot.png
"""
import os
import pandas as pd
import numpy as np
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

LIVING_COND_VARS = [
    "eau_potable", "electricite", "assainissement_ameliore",
    "mur_dur", "toit_dur", "sol_dur",
    "possede_radio", "possede_tv", "possede_refrigerateur",
    "possede_moto", "possede_voiture", "possede_telephone",
    "nb_pieces", "personnes_par_piece", "taille_menage",
]


def run_pca(n_components=5):
    df = pd.read_csv(IN_PATH)
    vars_present = [v for v in LIVING_COND_VARS if v in df.columns]
    if len(vars_present) < 3:
        raise ValueError(
            f"Trop peu de variables de conditions de vie trouvées ({vars_present}). "
            f"Vérifier les noms de colonnes dans menage_dataset.csv / data-dictionary EHCVM."
        )
    X = df[vars_present].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=min(n_components, len(vars_present)))
    scores = pca.fit_transform(Xs)

    explained = pca.explained_variance_ratio_
    print("Variance expliquée par composante :", np.round(explained, 3))
    print("Variance cumulée :", np.round(np.cumsum(explained), 3))

    # Variance plot
    plt.figure(figsize=(6, 4))
    plt.bar(range(1, len(explained) + 1), explained * 100)
    plt.plot(range(1, len(explained) + 1), np.cumsum(explained) * 100, color="red", marker="o")
    plt.xlabel("Composante")
    plt.ylabel("% variance expliquée")
    plt.title("ACP — Variance expliquée (conditions de vie)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "pca_variance.png"), dpi=150)
    plt.close()

    # Loadings (contribution des variables)
    loadings = pd.DataFrame(
        pca.components_.T, index=vars_present,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    loadings.to_csv(os.path.join(OUT_DIR, "pca_loadings.csv"))
    print("Variables les plus discriminantes sur PC1 :")
    print(loadings["PC1"].abs().sort_values(ascending=False).head(10))

    scores_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(pca.n_components_)])
    key_cols = [c for c in ["hhid", "region", "milieu"] if c in df.columns]
    scores_df = pd.concat([df[key_cols].reset_index(drop=True), scores_df], axis=1)
    scores_df.to_csv(os.path.join(OUT_DIR, "pca_scores.csv"), index=False)
    print(f"Scores ACP sauvegardés : {OUT_DIR}/pca_scores.csv")
    return scores_df, loadings, pca


if __name__ == "__main__":
    run_pca()
