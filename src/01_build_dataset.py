"""
01_build_dataset.py
====================
Fusionne les fichiers sources EHCVM 2018-2019 (Côte d'Ivoire) en un dataset ménage unique,
prêt pour l'ACP, le clustering, la classification et la régression.

Entrées attendues dans data/raw/ (format Stata .dta ou CSV, selon le téléchargement) :
    s00_me_civ2018, s01_me_civ2018, s02_me_civ2018, s03_me_civ2018, s04_me_civ2018,
    ehcvm_menage_CIV2018, ehcvm_welfare_CIV2018, ehcvm_ponderations_CIV2018

Sortie : data/processed/menage_dataset.csv

IMPORTANT (pauvreté vs quintile de richesse) :
- La variable cible officielle de pauvreté monétaire est construite ICI à partir de
  ehcvm_welfare_CIV2018 (dépense par équivalent-adulte comparée au seuil officiel).
- Le quintile de richesse (patrimoine/actifs) n'est PAS utilisé comme proxy de pauvreté ;
  il sert uniquement de variable descriptive complémentaire si disponible.
"""
import os
import pandas as pd
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)


def _read_any(basename):
    """Lit un fichier source quel que soit son format (.dta, .csv, .sav)."""
    for ext, reader in [(".dta", pd.read_stata), (".csv", pd.read_csv), (".sav", pd.read_spss)]:
        path = os.path.join(RAW_DIR, basename + ext)
        if os.path.exists(path):
            print(f"Lecture {path}")
            return reader(path, convert_categoricals=False) if ext == ".dta" else reader(path)
    raise FileNotFoundError(
        f"Fichier source introuvable: {basename}.(dta|csv|sav) dans {RAW_DIR}. "
        f"Voir data/raw/README.md pour la procédure de téléchargement."
    )


def build():
    menage = _read_any("ehcvm_menage_CIV2018")
    welfare = _read_any("ehcvm_welfare_CIV2018")
    poids = _read_any("ehcvm_ponderations_CIV2018")
    s00 = _read_any("s00_me_civ2018")
    s01 = _read_any("s01_me_civ2018")
    s02 = _read_any("s02_me_civ2018")
    s04 = _read_any("s04_me_civ2018")

    # Harmonisation de la clé ménage (nom typique EHCVM : grappe + menage, ou hhid)
    key_candidates = ["hhid", "grappe", "menage", "vague"]
    key = [c for c in key_candidates if c in menage.columns]
    merge_key = "hhid" if "hhid" in menage.columns else key

    df = menage.merge(welfare, on=merge_key, how="left", suffixes=("", "_welfare"))
    df = df.merge(poids, on=merge_key, how="left", suffixes=("", "_poids"))
    df = df.merge(s00, on=merge_key, how="left", suffixes=("", "_s00"))

    # Agrégation des caractéristiques du CHEF DE MÉNAGE depuis s01 (démographie) et s02 (éducation)
    if "lien_parente" in s01.columns:
        chef = s01[s01["lien_parente"] == 1].copy()  # 1 = chef de ménage (code EHCVM standard)
    else:
        chef = s01.copy()

    chef_cols = [c for c in ["sexe", "age", "situation_matrimoniale"] if c in chef.columns]
    chef_small = chef[[merge_key] + chef_cols] if isinstance(merge_key, str) else chef[merge_key + chef_cols]
    chef_small = chef_small.rename(columns={c: f"chef_{c}" for c in chef_cols})
    df = df.merge(chef_small, on=merge_key, how="left")

    # Éducation du chef (niveau atteint, alphabétisation) depuis s02
    edu_cols = [c for c in ["niveau_instruction", "alphabetisation", "diplome"] if c in s02.columns]
    if edu_cols and "individu_id" in s02.columns:
        edu = s02.merge(chef[[merge_key, "individu_id"]] if "individu_id" in chef.columns else chef[[merge_key]],
                         on=merge_key, how="inner")
        edu_small = edu[[merge_key] + edu_cols].rename(columns={c: f"chef_{c}" for c in edu_cols})
        df = df.merge(edu_small, on=merge_key, how="left")

    # Emploi du chef depuis s04
    emp_cols = [c for c in ["statut_emploi", "secteur_activite", "branche_activite"] if c in s04.columns]
    if emp_cols:
        emp_small = s04[[merge_key] + emp_cols].rename(columns={c: f"chef_{c}" for c in emp_cols})
        df = df.merge(emp_small, on=merge_key, how="left")

    # --- Cible officielle de pauvreté monétaire (seuil EHCVM) ---
    # Noms usuels dans ehcvm_welfare : 'pauvre' (0/1), 'dtot' ou 'depense' (dépense totale),
    # 'zref'/'seuil_pauvrete' (seuil), 'hhweight' (pondération).
    poverty_flag_col = next((c for c in ["pauvre", "poor", "pauvrete"] if c in df.columns), None)
    if poverty_flag_col:
        df["pauvre_officiel"] = df[poverty_flag_col]
    else:
        dep_col = next((c for c in ["dtot", "depense_totale", "depense_eq_adulte", "welfare"] if c in df.columns), None)
        seuil_col = next((c for c in ["zref", "seuil_pauvrete", "poverty_line"] if c in df.columns), None)
        if dep_col and seuil_col:
            df["pauvre_officiel"] = (df[dep_col] < df[seuil_col]).astype(int)
        else:
            print("ATTENTION : variable de pauvreté officielle non trouvée automatiquement — "
                  "vérifier les noms de colonnes de ehcvm_welfare_CIV2018 et ajuster ce script.")

    out_path = os.path.join(OUT_DIR, "menage_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"Dataset fusionné sauvegardé : {out_path} ({df.shape[0]} ménages, {df.shape[1]} variables)")
    return df


if __name__ == "__main__":
    build()
