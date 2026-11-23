# ML Air Quality Morocco

Projet de machine learning pour l'analyse et la modélisation temporelle de la qualité de l'air au Maroc.

## Contenu

- `Scripts/` : téléchargement, analyse exploratoire et entraînement des modèles temporels.
- `Data/README_pollution_weather.md` : description des sources de données.
- `Data/` : données locales et résultats générés (les fichiers CSV sont exclus du dépôt via `.gitignore`).

## Exécution

Installer les dépendances nécessaires, puis lancer les scripts depuis le dossier du projet :

```bash
python Scripts/download_city_pollution_weather.py
python Scripts/train_temporal_models.py
```

Le notebook d'analyse exploratoire se trouve dans `Scripts/morroco_city_analytics.ipynb`.

Les résultats CSV sont générés à l'exécution et ne sont pas versionnés.

## Dépôt

Ce projet est destiné au dépôt GitHub `Abdelkodous520/ml-air-quality-morocco`.
