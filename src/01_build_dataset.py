"""
01_build_dataset.py  -  Fusionne les fichiers EHCVM 2018-2019 (Côte d'Ivoire) en un dataset
ménage prêt pour l'ACP, le clustering, la classification et la régression.

Entrées (CSV, data/raw/) : ehcvm_menage_civ2018.csv, ehcvm_welfare_civ2018.csv
Sortie : data/processed/menage_dataset.csv (12 992 ménages)

IMPORTANT (pauvreté vs quintile de richesse) :
- La cible officielle de pauvreté monétaire est construite ICI à partir de
  ehcvm_welfare_civ2018 : pauvre_officiel = 1 si pcexp (dépense/équivalent-adulte) < zref (seuil officiel INS).
- Le quintile de richesse ('quintile' dans ehcvm_welfare) est un indice RELATIF basé sur le classement
  de pcexp en 5 groupes égaux  -  ce n'est PAS un indicateur de pauvreté absolue et n'est jamais utilisé
  comme cible dans ce projet.
"""
import os
import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)


def build():
    menage = pd.read_csv(os.path.join(RAW_DIR, "ehcvm_menage_civ2018.csv"))
    welfare = pd.read_csv(os.path.join(RAW_DIR, "ehcvm_welfare_civ2018.csv"))

    df = welfare.merge(menage, on="hhid", how="left", suffixes=("", "_m"))

    # Cible officielle de pauvreté monétaire (seuil EHCVM, INS)
    df["pauvre_officiel"] = (df["pcexp"] < df["zref"]).astype(int)

    # Variables de conditions de vie / équipement (0/1) dérivées de ehcvm_menage
    df["electricite"] = df["elec_ac"].apply(lambda x: 1 if x == 1 else 0)
    df["eau_potable"] = df["eauboi_ss"].apply(lambda x: 1 if x == 1 else 0)
    df["assainissement_ameliore"] = df["toilet"].apply(lambda x: 1 if x in [1, 2, 3] else 0)
    df["possede_tv"] = df["tv"].fillna(0).apply(lambda x: 1 if x == 1 else 0)
    df["possede_refrigerateur"] = df["frigo"].fillna(0).apply(lambda x: 1 if x == 1 else 0)
    df["possede_voiture"] = df["car"].fillna(0).apply(lambda x: 1 if x == 1 else 0)
    df["possede_ordinateur"] = df["ordin"].fillna(0).apply(lambda x: 1 if x == 1 else 0)

    out_path = os.path.join(OUT_DIR, "menage_dataset.csv")
    df.to_csv(out_path, index=False)

    taux = np.average(df["pauvre_officiel"], weights=df["hhweight"])
    print(f"Dataset fusionné : {out_path} ({df.shape[0]} ménages, {df.shape[1]} variables)")
    print(f"Taux de pauvreté officiel pondéré (national) : {taux*100:.1f}%")
    return df


if __name__ == "__main__":
    build()
