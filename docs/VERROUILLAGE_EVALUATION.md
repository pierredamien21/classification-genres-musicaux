# Verrouillage de l’évaluation finale — T6

Le verrouillage fixe les modèles et la méthode avant de calculer les scores test.
Il conserve les choix scientifiques du protocole préparatoire : neuf exports,
trois fenêtres fixes, moyenne des probabilités, bootstrap apparié par artiste,
aucun nouvel ajustement sur train + validation. La sensibilité au bruit reste une
limite acceptée après T2a/T2b ; elle n’est pas effacée par le verrouillage.

## Création et contrôles

Depuis la racine, avec l’environnement du projet :

```powershell
.\.venv\Scripts\python.exe scripts/lock_final_evaluation.py
```

Le script refuse de remplacer un verrouillage déjà publié. Il réunit les preuves
T1–T5, y compris T2a/T2b, et leurs contrôles indépendants. Toute empreinte historique
contradictoire ou modifiée provoque un échec. Il vérifie les neuf exports, leurs
rapports et caches, le checkpoint préentraîné, les sources Python, les partitions
et les empreintes des 7 857 fichiers audio retenus. Les artistes et groupes de
contenu doivent rester séparés entre train, validation et test.

Les fichiers test sont lus comme octets pour vérifier leur SHA-256 ; ils ne sont
ni décodés ni soumis aux modèles. Le chronométrage utilise seulement le plus petit
identifiant d’entraînement de chaque genre : huit morceaux, neuf modèles, sans
sélection selon leurs prédictions. Il conserve les durées par morceau et le coût
de chargement. L’extrapolation vers 778 morceaux est une estimation, avec une marge
de planification ×1 à ×2 ; elle exclut bootstrap, contrôles d’intégrité, rapport et
répétition à blanc sur validation.

Le résultat est un nouveau `reports/final-evaluation-lock-<date>.json`. Le rapport
préparatoire et sa configuration restent inchangés et gardent leur ancien statut.
La configuration y est incorporée comme provenance historique : l’autorisation
verrouillée se trouve au niveau supérieur du nouveau manifeste.

Le pointeur `reports/latest_final_evaluation_lock.json` n’est écrit qu’après
relecture complète du manifeste, nouvelle vérification de ses empreintes et
acceptation des identifiants de validation par le contrôle de partition. Le seul
fait qu’un fichier daté existe ne signifie donc pas que T6 est validée.

## Utilisation par T7

`load_locked_manifest(root, relative, expected_sha256)` vérifie le manifeste,
les fichiers requis et les versions des bibliothèques avant tout décodage.
`validate_evaluation_request(locked, track_ids, split)` exige exactement la
partition complète de validation ou de test, sans doublon ni mélange. Le digest
attendu doit venir du pointeur local conservé ; ce mécanisme détecte des dérives,
mais n’authentifie pas des fichiers face à une réécriture simultanée du manifeste
et du pointeur. Il ne remplace pas un environnement de travail stable pendant
l’exécution.

**T6 ne lance pas l’évaluation finale.** T7 doit encore implémenter l’évaluateur
reprenable, enregistrer son code et sa commande avec l’empreinte de ce verrouillage,
puis réussir une répétition à blanc sur validation avant le test. Les sources
épinglées ne doivent pas être modifiées silencieusement. Une modification impose
une nouvelle vérification et une décision explicitement tracée.

## Calcul en arrière-plan

Le lancement conserve ses métadonnées dans `reports/FINAL_LOCK_JOB.json`.
L’essai Windows détaché y enregistre un PID et les chemins des journaux ; la
relance en session persistante y enregistre son identifiant de session. Le script publie sa phase et ses compteurs dans
`reports/FINAL_LOCK_PROGRESS.json`, puis `complete` ou `failed`. Une interruption
brutale peut laisser `running` : vérifier alors aussi le processus et le journal.
Le PID seul ne prouve pas une réussite.

Ce traitement n’entraîne aucun nouveau modèle. Pour les prochaines campagnes
longues : communiquer une durée estimée, lancer sans bloquer la conversation,
conserver les résultats intermédiaires, puis consulter l’état sur demande ou avec
une surveillance espacée. Une surveillance par assistant peut consommer des tokens
à chaque réveil ; les journaux locaux continuent sans interaction avec l’assistant.

## Vérifications disponibles

Les tests de `tests/test_final_protocol.py` couvrent un verrouillage valide et les
refus : protocole fermé, empreintes modifiées, source absente, preuve incomplète,
version différente, chemin hors projet, modèles incohérents, revue non résolue,
fuite entre partitions, labels ou identifiants incorrects et requêtes partielles.
Les résultats effectifs de la campagne T6 sont suivis dans
[le résumé de partie](resumes/31_VERROUILLAGE_FINAL.md).
