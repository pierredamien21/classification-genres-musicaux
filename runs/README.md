# Expériences

Créer un dossier par exécution avec un identifiant unique :
`YYYYMMDD_HHMMSS_nom_experience_seed`, complété pour éviter les collisions.

Chaque exécution conservera : configuration résolue, environnement, version du code, manifeste et checksum, graine, données de préentraînement connues, logs, métriques validation, temps, mémoire et statut final. Les mesures test seront distinctes et produites seulement après verrouillage.

Distinguer terminé, interrompu, échoué et arrêté pour budget ; ne pas écraser une expérience. En l’absence de Git, enregistrer une empreinte des sources. Les rapports synthétiques sélectionnés iront dans `reports/`. Aucun entraînement n’a encore été lancé.

