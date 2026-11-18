# Brouillon post LinkedIn (à valider avant publication)

🔍 **Peut-on prédire la vulnérabilité des ménages ivoiriens à partir du Machine Learning ?**

J'ai mené une analyse multidimensionnelle de la pauvreté en Côte d'Ivoire à partir des
microdonnées officielles EHCVM 2018-2019 (INS / Banque Mondiale) — 12 992 ménages, 61 116 individus.

🎯 Objectif : identifier les profils de ménages les plus vulnérables à partir de leurs
conditions de vie, niveau d'éducation, situation professionnelle et dépenses — sans jamais
confondre quintile de richesse (patrimoine) et pauvreté monétaire officielle (seuil EHCVM).

📊 Méthodologie :
• ACP pour réduire les dimensions des conditions de vie
• K-Means pour construire une typologie de ménages
• Classification supervisée pour prédire le statut de pauvreté officiel
• Régression pondérée pour expliquer le niveau de vie
• Cartographie des disparités régionales

📌 Résultats clés : [à compléter après exécution réelle sur les données]
• Taux de pauvreté par région
• Typologie de ménages (urbain/rural)
• Variables les plus discriminantes de la vulnérabilité
• Recommandations pour le ciblage des politiques sociales

💻 Code et méthodologie complète en open-source :
🔗 https://github.com/MediaBakayoko/ehcvm-poverty-ml

#DataScience #MachineLearning #Pauvreté #CôtedIvoire #EHCVM #GeoAI #SuiviEvaluation #PolitiquesSociales

---
NOTE: Les chiffres [à compléter] seront remplacés par les résultats réels une fois le pipeline
exécuté sur les microdonnées (en attente de confirmation du compte World Bank Microdata).
