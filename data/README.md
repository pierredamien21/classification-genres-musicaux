# Données locales

- `raw/` : archives, audio FMA et métadonnées originales, conservés sans modification.
- `manifests/` : inventaires, partitions, exclusions et empreintes.
- `interim/` : résultats temporaires ou intermédiaires recréables.
- `features/` : descripteurs, spectrogrammes et embeddings mis en cache.

Les fichiers de ces dossiers sont exclus de Git. Pour publier une expérience, exporter un manifeste nettoyé et versionné avec un checksum dans ses artefacts de reproduction ; un chemin local ignoré ne suffit pas à la reproduire.

## Acquisition officielle FMA

Acquisition terminée le 20 octobre 2026 via `scripts/acquire_fma.py` (bibliothèque standard Python + curl). Les SHA-1 officiels et les CRC sont vérifiés ; les SHA-256 sont enregistrés. Le script conserve les `.part` pour reprise lorsqu’un futur téléchargement est interrompu.

- `raw/archives/fma_small.zip` et `raw/archives/fma_metadata.zip` : archives vérifiées.
- `raw/fma_small/` : fichiers audio rangés dans les sous-dossiers FMA d’origine.
- `raw/fma_metadata/` : CSV et fichiers livrés par les auteurs.
- `manifests/acquisition_fma.json` : état de l’acquisition, provenance et contrôles.
- `manifests/fma_small_inventory.json` : correspondance vérifiée des 8 000 MP3 avec les métadonnées, classes et partitions.

L’inventaire est reproductible avec `scripts/check_fma_inventory.py` : aucun fichier manquant/inattendu, huit genres de 1 000 morceaux, partitions 6 400/800/800, aucun identifiant d’artiste partagé entre partitions. Source : [dépôt officiel FMA](https://github.com/mdeff/fma).

Les licences audio dépendent des morceaux ; ne pas attribuer automatiquement à tout l’audio la licence des métadonnées.

## Audit audio

```powershell
.\.venv\Scripts\python.exe scripts/audit_fma.py
.\.venv\Scripts\python.exe scripts/summarize_audit.py
```

Chaque audit crée un nouveau dossier `manifests/audit-<date UTC>/` ; aucun fichier audio
n’est modifié. `latest_audit.json` pointe vers le dernier audit terminé et en conserve
l’empreinte du manifeste. Un audit interrompu ne remplace pas ce pointeur.

- `manifest.csv` : chaque morceau, genre, artiste, partition, licence, empreintes, métriques et statut.
- `report.json` : configuration, versions, empreintes du code et métadonnées, comptages et doublons.
- `decoder-*.log` : diagnostics natifs encadrés par identifiants de morceaux.
- `results-*.jsonl` : résultats bruts par processus ; un échec reste traçable.
- `review_queue.csv` : morceaux à examiner et exclusions techniques, exportés par la synthèse.
- `reports/AUDIT_DONNEES.md` : synthèse lisible, créée après vérification de l’empreinte du manifeste.

Le parallélisme utilise des processus isolés pour attribuer les messages natifs aux fichiers.
Il ne nécessite pas de GPU. La détection de doublons exacts par fichier et signal PCM
ne remplace pas une recherche de quasi-doublons par empreintes perceptuelles.
Les entrées `needs_review` attendent une décision explicite avant l’entraînement.

## Sélection pour les premières expériences

La revue automatisée est disponible : voir
[`docs/REPRODUCTION_REVUE.md`](../docs/REPRODUCTION_REVUE.md) et
[`reports/REVUE_DONNEES.md`](../reports/REVUE_DONNEES.md).
`manifests/latest_baseline.json` indique le manifeste versionné à utiliser
(7 857 lignes), son empreinte et le journal complet des 8 000 décisions.
Les 126 cas en quarantaine restent réexaminables. Les 17 exclusions de cette
sélection regroupent six échecs de décodage et onze répétitions exactes.
L’audit initial conserve ses statuts historiques ; le journal de revue est distinct.

## Caractéristiques musicales

`features/latest_descriptors.json` pointe vers la première extraction train/validation :
21 237 fenêtres × 125 valeurs, avec `features.npy`, `index.csv`, configurations et empreintes.
Les morceaux de test ne sont pas décodés par cette extraction. Le cache par morceau
permet une reprise ; toute anomalie interrompt le traitement sans exclusion implicite.
Voir [`docs/DESCRIPTEURS_MUSICAUX.md`](../docs/DESCRIPTEURS_MUSICAUX.md) pour reproduire
l’extraction et retrouver l’expérience de référence.

