# OpenClassrooms parcours DATA SCIENTIST
## Implémentez un modèle de scoring

### Objectifs:
- Entrainer un modèle de scoring à partir de données clients ayant contracté des prêts bancaires. 
Les clients sont labellisés 0 (bon payeur) ou 1 (défaut de paiement)
- Implémenter le modèle dans une application web explicant le score

# Prétraitement:

- Utilisation du script de AGUIAR sur KAGGLE : https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features

    [Voir le script](kaggle_kernel_raw.py)

- Les données sont déséquilibrées (seulement 8% de positifs). j'ai donc essayé de faire un clustering des negatifs
(k-means) afin d'en dégager des catégories. Sans succès. j'ai donc décidé d'utiliser un RandomUnderSampler 
par volonté de faire simple et de réduire la taille du dataset. Les resultats de detection des positifs sont 
sensiblement améliorés (calcul du f1_score sur la classe 1). 

    [Voir le script](target_balance.py)

# Modélisation

- Entrainement de modèle avec GridSearch et CrossValidation sur le set resampled (70% des données). 
f1_score de la classe 1 comme score de référence f1_score de 0.6 sur la classe 1 avec CV sur les données train, 
re-échantillonnées

[Voir le script](modeling.py)


- Prediction sur le set de test (30% de données). Obtention f1_score de la classe 1 sur le set sans rééchantillonage : 0.29
- Creation d'un score de perte en montant avec des valeurs arbitraires de pertes en fonction de l'erreur.

[Voir fonction 'Loss Score'](udf.py)

- Determination des seuils en se basant sur les valeurs de pertes. Rappel: n'étant pas banquier, je ne peux garantir que 
les seuils soient cohérents avec une vision métier, la fonction de perte ainsi que les seuils doivent être réalisés
avec un expert bancaire

[Voir le script](threshold.py)

# Dashboard

Le dashboard est réalisé avec le script [dash_board.py](dash_board.py) qui fonctionne avec le framework streamlit.
Streamlit lit le fichier via github. Ensuite, les données sont chargées en cache via AWS s3. Les prédictions et 
les explications du modèle sont générées via des requêtes POST vers une [API](app.py) hébergée sur heroku qui accède aux 
scripts et au modèle via github également.

Le dashboard présente donc le score prédit par le modèle et les variables les plus explicatives. 
Par ailleurs, on dispose aussi d'un histogramme permettant de visualiser la repartition d'une variable et la postition 
du client ciblé par rapport aux autres
