# Sensibilité au volume et au bruit — panel de validation

64 morceaux, huit par genre et 64 artistes distincts. Ce panel fixé avant les essais n’est pas l’ensemble des 780 morceaux de validation. Test non évalué.

| Condition | Macro-F1 | Accuracy | Perte F1 vs identité | Classes changées | Corrections / régressions | Revue déclenchée |
|---|---:|---:|---:|---:|---:|---|
| identity | 0.5479 | 0.5469 | +0.0000 | 0/64 | 0 / 0 | non |
| gain_minus_6_db | 0.5050 | 0.5000 | +0.0429 | 6/64 | 1 / 4 | non |
| white_noise_snr_20_db | 0.3648 | 0.4062 | +0.1831 | 26/64 | 1 / 10 | oui |

Seuils prévus : perte macro-F1 ≥ 0,05 ou changements de classe ≥ 20 %. Une revue déclenchée ne lance aucun ajustement du modèle.

| Genre | F1 identité | F1 gain −6 dB | F1 bruit 20 dB | Changements gain / bruit |
|---|---:|---:|---:|---:|
| Electronic | 0.6316 | 0.4706 | 0.5000 | 3 / 1 |
| Experimental | 0.5333 | 0.4706 | 0.3333 | 0 / 3 |
| Folk | 0.5556 | 0.5263 | 0.4000 | 0 / 6 |
| Hip-Hop | 0.7059 | 0.7500 | 0.5714 | 0 / 0 |
| Instrumental | 0.2857 | 0.3750 | 0.0000 | 1 / 7 |
| International | 0.6667 | 0.6667 | 0.3636 | 0 / 5 |
| Pop | 0.2353 | 0.2353 | 0.1250 | 0 / 2 |
| Rock | 0.7692 | 0.5455 | 0.6250 | 2 / 2 |

Les fichiers d’origine sont conservés. Transformations sur mono 16 kHz avant segmentation, padding intact, passage WAV FLOAT vérifié sans perte, même API et même modèle. Ni normalisation supplémentaire ni écrêtage ajouté.

Benchmark : projection 8.00 minutes sous le budget de 120 minutes. Exécution complète : 7.46 minutes, un processus ; mesure indicative sur ce poste.

Avec huit morceaux par genre, ces scores sont descriptifs. Ils n’établissent pas une robustesse générale ni une cause acoustique certaine. Aucun intervalle de confiance ou test de significativité n’est revendiqué.

Reproduction : `.\.venv\Scripts\python.exe scripts/evaluate_robustness.py`.

[Résultats détaillés et empreintes](ROBUSTNESS_PANEL.json) · [Protocole](../docs/archive/PROTOCOLE_EVALUATION_FINALE.md)
