# Reproduire la revue et la sélection initiale

Le manifeste de départ contient les 8 000 morceaux audités. La revue ajoute une
empreinte perceptuelle locale et applique le profil `conservative_baseline_v1`.
Elle ne modifie jamais le MP3 original ou sa partition.

## Commandes

Depuis la racine du projet, après l’audit :

```powershell
.\.venv\Scripts\python.exe scripts/fetch_chromaprint.py
.\.venv\Scripts\python.exe scripts/validate_fingerprints.py
.\.venv\Scripts\python.exe scripts/scan_fingerprints.py
.\.venv\Scripts\python.exe scripts/build_baseline_manifest.py
.\.venv\Scripts\python.exe scripts/summarize_review.py
```

La première commande utilise l’archive officielle de Chromaprint 1.6.2, dans
`data/interim/tools/`, sans installation globale. L’archive et l’exécutable sont
vérifiés avec les SHA-256 de `configs/review_policy.json`. Le SHA-256 de l’archive
correspond au digest publié pour l’asset GitHub officiel lors de l’acquisition.
Les autres commandes n’ont besoin d’aucun accès réseau.

L’exécutable Windows x86-64 fonctionne sur ce poste ARM64. Il embarque un décodeur
FFmpeg distinct de libsndfile/mpg123. Son succès signifie qu’une empreinte a pu être
calculée, pas que le signal est sans défaut. La politique conserve les avertissements
du décodeur de référence même si fpcalc termine normalement.

## Artefacts et provenance

- `latest_audit.json` : audit de référence et empreinte du manifeste.
- `latest_fingerprints.json` : dernier scan achevé, rapport et empreinte.
- `latest_baseline.json` : dernière sélection achevée, empreintes du manifeste et du rapport.

Ces pointeurs se trouvent dans `data/manifests/`. Chaque exécution crée un nouveau
dossier daté. Le constructeur contrôle les empreintes avant de produire :

- `manifest.csv` : seuls les morceaux retenus, prêts pour les expériences initiales.
- `decisions.csv` : les 8 000 lignes, avec statut d’audit, décision, motifs,
  représentant éventuel et identifiant de groupe de contenu.
- `report.json` : politique, effectifs, versions de sources et provenance.

La synthèse ajoute `pending_review.csv` et `fingerprint_candidates.csv` et produit
[`reports/REVUE_DONNEES.md`](../../reports/REVUE_DONNEES.md).
Les décisions sont `keep`, `hold` ou `exclude`. `hold` est une quarantaine réversible.
`exclude` peut désigner une répétition redondante, pas uniquement un fichier défectueux.
La paire 057371 / 072149 est isolée pour conflit de genres sur un signal exact.

## Portée scientifique

Le profil de décision v1 est implémenté dans `src/music_genre/data/review.py`.
La section `decisions` de la configuration en décrit les règles ; ce n’est pas
un langage de programmation de règles. Modifier une décision nécessite une nouvelle
version du profil, une modification du code et des vérifications adaptées.
Les seuils numériques d’empreintes sont effectivement lus depuis la configuration.

La recherche de candidats utilise des mots Chromaprint exacts pour proposer des paires,
puis une distance de Hamming après alignement. Les candidats non exacts et leur groupe
associé sont isolés, sans affirmer que chacun est un doublon. Les seuils ont été fixés
avant l’analyse des 8 000 morceaux et n’ont pas été adaptés aux résultats du test.

Les contrôles comprennent les douze paires PCM connues et six transformations de trois
sources de training. Ils ne constituent pas une estimation du rappel sur tous les
quasi-doublons possibles. Changements importants, recouvrements courts et extraits
non recouvrants d’un même morceau peuvent échapper à la méthode.

La sélection contient 6 299 train, 780 validation et 778 test. Les performances futures
devront être rattachées à son empreinte et à ces effectifs : ce protocole filtré ne
permet pas une comparaison directe avec les scores publiés sur FMA-small non filtré.
Une analyse de sensibilité des mises à l’écart reste envisageable après une revue
supplémentaire ; ne pas sélectionner les exclusions selon le score obtenu.

Références : [Chromaprint, objectifs et limites](https://github.com/acoustid/chromaprint),
[fpcalc et calcul local des empreintes](https://acoustid.org/chromaprint).
