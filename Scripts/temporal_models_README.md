# Comparer des modèles temporels

Le script `train_temporal_models.py` compare une régression linéaire, une
Random Forest, XGBoost (si `xgboost` est installé) et une référence naïve qui
reprend la mesure actuelle comme prévision de l'heure suivante.

## Lancer le script

Depuis le dossier `Projet` :

```powershell
python Scripts\train_temporal_models.py
```

Dépendances requises : `pandas`, `numpy` et `scikit-learn`. XGBoost est
facultatif ; pour l'inclure :

```powershell
pip install pandas numpy scikit-learn xgboost
```

Le CSV par défaut est `Data\morocco_city_pollution_weather_hourly.csv`. Les
résultats sont écrits dans `Data\temporal_model_comparison.csv` et
`Data\temporal_model_predictions.csv`.

## Logique de préparation

- Le script trie les observations par ville (ou station) et par date.
- Il crée le PM2.5 courant (`lag_0h`), ainsi que les valeurs précédentes d'une,
  trois et vingt-quatre heures.
- La cible `target_next_hour` est la concentration une heure après la ligne.
- Les retards et la cible sont invalidés si les horodatages ne sont pas espacés
  comme attendu. Les lignes avec valeurs manquantes sont écartées.
- Les dernières dates sont gardées pour le test ; les dates antérieures servent
  à l'entraînement. Aucun mélange aléatoire des heures n'est effectué.
- MAE et RMSE plus faibles indiquent de meilleures erreurs ; un R² plus élevé
  indique une meilleure explication des variations.

## Utiliser des mesures de station

Le CSV fourni par défaut contient des polluants estimés par CAMS, pas des
observations de stations. Pour utiliser des mesures, prépare un CSV horaire
avec au minimum une colonne ville ou station, une colonne date et heure, et la
colonne cible. Les variables météo reconnues par le script sont
`temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `precipitation`,
`surface_pressure`, `cloud_cover`, `wind_speed_10m`, `wind_direction_10m`,
`wind_gusts_10m` et `shortwave_radiation`.

Par exemple, si ton fichier a `station_id`, `timestamp` et une cible nommée
`PM2_5`, indique les noms ainsi :

```powershell
python Scripts\train_temporal_models.py `
  --input Data\mesures_horaires.csv `
  --group station_id `
  --datetime timestamp `
  --target PM2_5
```

L'évaluation mesure une prévision à une heure. Si tu ajustes les paramètres des
modèles, fais ce réglage avec une validation chronologique distincte et garde
la période de test finale intacte. Les variables de trafic ne sont pas encore
incluses : ajoute-les au CSV horaire et à la liste des variables du script si
tu obtiens une source avec une couverture suffisante.
