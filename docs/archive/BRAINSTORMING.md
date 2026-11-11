# Brainstorming critique — ce qu’il ne faut pas oublier

20 octobre 2026. Relecture du plan et de l’architecture. Les mentions « prévu » ou « décidé » décrivent le cadrage, pas une protection déjà implémentée. Il n’est pas possible de garantir qu’il ne reste aucun oubli ; cette liste sert de registre de vérification.

## 1. Avant toute extraction complète

| Point | Pourquoi il compte | Action retenue |
|---|---|---|
| Python ARM64 / x86-64 | Le système et le processus Python n’ont pas la même architecture | Vérifier imports, décodage et mesure de temps ; ne pas déduire les performances des 16 Go de RAM |
| Dépendances et codecs | Une installation réussie ne garantit pas la lecture des MP3 | Essayer WAV et MP3 réels dans un environnement isolé avant de verrouiller les versions |
| Espace disque | Archives, extraction et caches peuvent coexister et dépasser largement le seul téléchargement | Estimer la taille à partir d’un petit échantillon ; conserver les caches sous contrôle |
| Provenance des données | Les originaux et exclusions doivent rester traçables | Contrôles d’intégrité, manifeste et journal d’audit |
| Fuites de données | Un même morceau, artiste ou doublon peut gonfler les scores | Vérifier les partitions avant segmentation ; garder ensemble les fenêtres d’un morceau |
| Alignement de la fusion | Deux matrices peuvent être valides mais associées aux mauvais morceaux | Jointure explicite sur track_id et window_id |
| Reproductibilité | Un cache ignoré par Git peut devenir introuvable | Publier une recette et les empreintes du manifeste, des configurations et des poids |

Observation locale : le socle fonctionne avec Python 3.13 x86-64, mais la compatibilité de la pile audio/ML n’a pas été validée. Le diagnostic ne mesure pas encore la RAM disponible ou les capacités de calcul.

## 2. Définition des genres

Les catégories « Instrumental », « International » ou « Experimental » n’ont pas toutes la même signification musicale. Elles peuvent être ambiguës et se recouper conceptuellement avec d’autres classes. L’objectif est d’apprendre les étiquettes de FMA-small, pas de définir une vérité universelle du genre.

Prévoir une fiche du dataset indiquant couverture, langues si connues, équilibre par classe et nombre d’artistes. Ne pas renommer Hip-Hop en rap sans justification. Ne pas promettre de reconnaître le gospel alors qu’il n’est pas une classe apprise.

Le système est fermé à huit classes. Il pourra donner un score élevé à un genre absent ou à un fichier non musical. La première version doit exposer cette limite ; la détection de hors-distribution et un mode d’abstention nécessitent un protocole propre, pas un seuil arbitraire choisi sur quelques exemples.

## 3. Prétraitement : détails encore à figer

Le premier plan fixait FFT, pas et nombre de bandes Mel, mais il manque encore : fenêtre d’analyse, centrage, puissance du spectre, f_min/f_max, définition du logarithme, seuil numérique et normalisation d’entrée du ViT. Ces choix influencent le résultat ; les figer après le notebook audio et avant les expériences comparatives.

Distinguer trois opérations : normalisation du signal ; préparation des entrées du checkpoint ; standardisation des descripteurs/embeddings pour le classificateur. Elles n’ont pas le même rôle. Toute statistique calculée sur notre dataset doit être ajustée sur train seulement.

Autres points à vérifier :
- conversion stéréo vers mono susceptible d’annuler certaines composantes en opposition de phase ;
- anti-repliement lors du rééchantillonnage ;
- silence, faibles niveaux et valeurs non finies sans assimiler automatiquement une musique calme à une erreur ;
- padding et durée valide pour ne pas faire voter des fenêtres entièrement silencieuses ajoutées ;
- compromis 16 kHz et pertes de contenu aigu ;
- paramètres du HPSS et artefacts de séparation ;
- ambiguïtés tempo double/moitié, choix de la grille rythmique et stabilité des estimations ;
- mémoire au décodage d’un fichier compressé, indépendamment de sa taille sur disque.

Ces sujets doivent produire des exemples écoutables et des décisions explicites, pas une multiplication automatique des transformations.

## 4. Comparaisons et apport personnel

Expérience centrale conservée : descripteurs seuls / ViT seul / fusion. Une amélioration de fusion peut venir du nombre de dimensions ou du classificateur : garder la même famille et le même budget de réglage, puis retirer les groupes rythme et harmonie pour interpréter le gain.

Le CNN entraîné de zéro et le ViT préentraîné ne diffèrent pas seulement par l’architecture. Le rapport doit le dire. L’AST change aussi préentraînement, taille et prétraitement. Éviter les conclusions causales trop larges.

Prévoir un contrôle de bon fonctionnement du CNN sur un très petit lot avant un entraînement long. Ne pas en tirer une performance de généralisation.

Les ablations doivent être décidées à l’avance. Choisir sur validation les variantes à comparer, puis ouvrir le test une fois les décisions figées. Revenir sur le test pour chaque amélioration transformerait progressivement ce test en validation.

Les comparaisons avec des modèles musicaux externes devront documenter l’exposition possible aux morceaux du test pendant leur préentraînement, même sans étiquettes. La revue identifie déjà MusicFM-FMA comme un cas à exclure du classement principal.

## 5. Interprétation et incertitude

Ajouter une analyse des erreurs par artiste, classe et qualité du signal. Une classe abondante en morceaux peut être dominée par peu d’artistes. Les intervalles de confiance seront calculés par groupes d’artistes.

Si les trois fenêtres d’un morceau se contredisent, conserver cette information pour l’analyse et éventuellement la montrer dans la démo. Une moyenne peut masquer cette variation.

Les cartes d’attention ou masquages sont des outils d’exploration. Ils ne prouvent pas que le réseau a identifié un instrument, compris le rythme ou utilisé une cause musicale.

Les scores ne sont pas automatiquement calibrés. Une future calibration doit utiliser des données réservées à cet effet et être évaluée ; elle ne doit pas conduire à affaiblir le test final.

## 6. Ingénierie et portfolio

Décisions déjà intégrées à l’architecture :
- même prédiction pour CLI et interface ;
- classes ordonnées et sauvegardées avec le modèle ;
- séparation checkpoints de reprise / exports de prédiction ;
- modules importables sans téléchargement implicite ;
- conservation des échecs et interruptions dans les expériences ;
- caches invalidables et sources traçables ;
- documentation de ce qui est réellement implémenté.

À prévoir avant livraison :
- vérifier licences des données, poids, exemples audio et choisir la licence du code ;
- ne pas versionner de secrets ou de chemins privés dans les résultats publics ;
- rendre l’export utilisable sans le dataset original ;
- tester depuis un environnement neuf et documenter les limites Windows ARM64 ;
- mesurer le temps à froid et à chaud séparément pour la démo ;
- supprimer les fichiers déposés après traitement ;
- fournir une vidéo montrant aussi une erreur réelle ;
- conserver des exemples de décisions scientifiques, résultats négatifs compris.

## 7. Priorités pour les prochaines séances

**P0 — Débloquer le travail :** environnement audio/ML, budget mesuré, espace disque, acquisition vérifiée, audit et partitions.

**P1 — Rendre l’expérience fiable :** conventions audio complètes, contrats et caches, baseline musicale, ViT gelé, fusion, tests d’intégration et protocole de validation.

**P2 — Améliorer si les mesures le justifient :** HPSS, couches intermédiaires, affinage, AST, explications et calibration. Ne pas ajouter tout ce niveau au périmètre obligatoire.

**Hors première version :** reconnaissance générale de musique inconnue, paroles, multi-étiquette, gospel collecté manuellement, modèle de fondation entraîné de zéro, infrastructure de déploiement complexe.

La prochaine séance doit aboutir à un petit diagnostic pratique : lire un WAV et un MP3, extraire une représentation, charger le ViT et chronométrer une passe. C’est le jalon permettant de fixer l’environnement et le budget avant de traiter tout FMA.

