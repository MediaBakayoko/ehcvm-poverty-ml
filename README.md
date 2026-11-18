# Machine Learning for Multidimensional Poverty Profiling in Côte d'Ivoire

## Problématique
Peut-on identifier les profils de ménages les plus vulnérables à partir de leurs conditions de vie,
de leur niveau d'éducation, de leur situation professionnelle et de leurs dépenses ?

## Source des données
**Enquête Harmonisée sur les Conditions de Vie des Ménages (EHCVM) 2018-2019**, Côte d'Ivoire,
produite par l'Institut National de la Statistique (INS), diffusée par la Banque Mondiale.

- Catalogue : https://microdata.worldbank.org/catalog/4292
- Téléchargement (inscription requise) : https://microdata.worldbank.org/catalog/4292/get-microdata
- Dictionnaire des variables : https://microdata.worldbank.org/catalog/4292/data-dictionary

Échantillon : **12 992 ménages** et **61 116 individus**.

### Fichiers sources utilisés

| Fichier | Contenu | Utilisation |
|---|---|---|
| `s00_me_civ2018` | Identification des ménages | Localisation |
| `s01_me_civ2018` | Composition et caractéristiques des membres | Démographie |
| `s02_me_civ2018` | Éducation | Capital humain |
| `s03_me_civ2018` | Santé | Accès aux soins |
| `s04_me_civ2018` | Emploi | Situation professionnelle |
| `ehcvm_menage_CIV2018` | Données des ménages | Analyse socioéconomique |
| `ehcvm_welfare_CIV2018` | Indicateurs de bien-être (dépenses, pauvreté) | Mesure du niveau de vie |
| `ehcvm_ponderations_CIV2018` | Pondérations | Représentativité statistique |

> ⚠️ Les microdonnées brutes **ne sont pas versionnées** dans ce dépôt (accès sous licence,
> inscription nominative requise auprès de la Banque Mondiale). Seuls le code, les données
> agrégées/anonymisées dérivées et les résultats sont publiés. Voir `data/raw/README.md`
> pour la procédure de téléchargement.

## Objectifs
1. Construire des profils socioéconomiques de ménages.
2. Mesurer les disparités entre régions.
3. Analyser les liens entre instruction, emploi et niveau de vie.
4. Identifier les caractéristiques associées à la vulnérabilité.
5. Comparer les profils urbains et ruraux.

## Méthodologie — précision méthodologique clé
La **pauvreté monétaire** est définie officiellement par l'INS/Banque Mondiale à partir de la
**dépense de consommation par équivalent-adulte** comparée au **seuil national de pauvreté**
(variable `pauvre`/`welfare` du fichier `ehcvm_welfare_CIV2018`, pondérée par `hhweight`).
Le **quintile de richesse** (basé sur un indice d'actifs/ACP patrimoniale, à la Wealth Index DHS)
est un indicateur **différent**, relatif et non monétaire. Les deux ne doivent **jamais être confondus** :
- Pauvreté monétaire → seuil officiel nationale, taux de pauvreté (headcount ratio) comparable aux chiffres INS.
- Quintile de richesse → classement relatif de l'échantillon en 5 groupes égaux, utile pour la
  segmentation mais pas pour mesurer un taux de pauvreté.

Dans ce projet, la cible supervisée "pauvre / non pauvre" provient **exclusivement** de
`ehcvm_welfare_CIV2018` (seuil officiel), jamais d'un quintile de richesse construit en ACP.

### Pipeline
1. **Nettoyage & fusion** (`src/01_build_dataset.py`) : fusion ménage + membres + welfare + pondérations
   sur l'identifiant ménage, agrégation des variables individuelles (éducation, emploi) au niveau ménage.
2. **ACP** (`src/02_pca.py`) : réduction des dimensions sur les variables de conditions de vie
   (équipements, logement, accès eau/électricité/assainissement, éducation du chef de ménage, etc.).
3. **K-Means** (`src/03_kmeans.py`) : segmentation en typologies de ménages à partir des composantes ACP.
4. **Classification supervisée** (`src/04_classification.py`) : prédiction du statut de pauvreté
   officiel (seuil EHCVM) — Random Forest / Gradient Boosting, avec pondération `hhweight`.
5. **Régression** (`src/05_regression.py`) : modélisation du log de la dépense par équivalent-adulte
   en fonction des caractéristiques du ménage (éducation, emploi, région, milieu).
6. **Cartographie** (`src/06_mapping.py`) : taux de pauvreté et typologies par région.

## Résultats produits (`reports/`)
- Carte des disparités régionales de pauvreté.
- Typologie des ménages (clusters K-Means).
- Profils des ménages vulnérables (caractéristiques discriminantes).
- Comparaison urbain/rural.
- Importance des variables (classification + régression).
- Recommandations de ciblage des politiques sociales.

## Reproductibilité
```bash
pip install -r requirements.txt
python src/01_build_dataset.py
python src/02_pca.py
python src/03_kmeans.py
python src/04_classification.py
python src/05_regression.py
python src/06_mapping.py
```

## Auteur
Média Marcel Bakayoko — Expert géomatique / données géospatiales & agricoles / GeoAI / Suivi-Évaluation.

## Licence des données
Microdonnées EHCVM 2018-2019 © Institut National de la Statistique de Côte d'Ivoire / Banque Mondiale,
diffusées sous licence d'accès pour la recherche (inscription nominative requise, usage non commercial).
