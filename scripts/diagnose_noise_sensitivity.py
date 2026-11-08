"""Replay the eight fixed pilots and localize representation drift without fitting."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from check_candidate_inference import read_csv

from music_genre.audio.processing import AudioBuffer, load_audio, segment_audio
from music_genre.data.audit import sha256_file, write_json
from music_genre.evaluation.noise_diagnostic import representation_drift
from music_genre.evaluation.robustness import perturb_audio
from music_genre.inference.predict import CandidatePredictor

ROOT = Path(__file__).resolve().parents[1]


def main():
    started = perf_counter()
    pointer = json.loads((ROOT / 'reports/latest_robustness_panel.json').read_text('utf-8'))
    path = ROOT / pointer['report']
    if sha256_file(path) != pointer['report_sha256']:
        raise ValueError('Robustness report changed')
    previous = json.loads(path.read_text('utf-8'))
    hashes = dict(previous['source_and_input_sha256'])
    for source, digest in hashes.items():
        if sha256_file(ROOT / source) != digest:
            raise ValueError(f'Previous input/source changed: {source}')
    for source in ('scripts/diagnose_noise_sensitivity.py', 'src/music_genre/evaluation/noise_diagnostic.py', pointer['report']):
        hashes[source] = sha256_file(ROOT / source)
    protocol = json.loads((ROOT / 'configs/final_evaluation_protocol.json').read_text('utf-8'))
    if protocol['test_evaluation_allowed'] or previous['test_evaluated']:
        raise ValueError('Diagnostic restricted to validation before test')
    prepared = json.loads((ROOT / 'reports/PREPARATION_EVALUATION_FINALE.json').read_text('utf-8'))
    cohort = {r['track_id']: r for r in read_csv(ROOT / prepared['sample']['path'])}
    ids = previous['benchmark']['track_ids']
    if len(ids) != 8 or len(set(ids)) != 8 or any(cohort[k]['split'] != 'validation' for k in ids):
        raise ValueError('Expected eight fixed validation pilots')
    stored = {r['track_id']: r for r in previous['results']}
    predictor = CandidatePredictor(ROOT)
    if predictor.model_id != previous['model_id']:
        raise ValueError('Model changed')
    config, model = predictor.config, predictor.model
    conditions = previous['conditions']
    labels = predictor.labels
    records, all_features = [], []
    max_error = 0.
    for key in ids:
        row = cohort[key]
        audio_path = (ROOT / 'data/raw' / row['audio_path']).resolve()
        if (not audio_path.is_relative_to((ROOT / 'data/raw/fma_small').resolve())
                or sha256_file(audio_path) != row['source_sha256']):
            raise ValueError('Changed or invalid source')
        audio = load_audio(audio_path)
        track = {'track_id': key, 'genre': row['genre'], 'conditions': {}}
        track_features = []
        original_class = int(np.argmax(stored[key]['conditions']['identity']['prediction']['scores']))
        for condition in conditions:
            samples = perturb_audio(audio.samples, condition, key)
            windows = segment_audio(AudioBuffer(samples, 16000, 16000, 1), tuple(config['offsets_seconds']), config['window_seconds'])
            mels, vectors = [], []
            for window in windows:
                if perf_counter() - started > 600:
                    raise RuntimeError('Ten-minute diagnostic budget exhausted')
                mel = predictor.extractor.spectrograms(window)
                vector = predictor.extractor.encoder.extract(mel, config['db_min'], config['db_max'])
                mels.append(mel)
                vectors.append(vector)
            vectors = np.stack(vectors)
            probs = model.predict_proba(vectors)
            scores = probs.mean(axis=0)
            old = stored[key]['conditions'][condition]['prediction']
            expected = np.array([w['scores'] for w in old['windows']])
            if (expected.shape != probs.shape or not np.allclose(probs, expected, atol=1e-6, rtol=0)
                    or not np.allclose(scores, old['scores'], atol=1e-6, rtol=0)):
                raise ValueError('Diagnostic does not reproduce the T2 inference path')
            max_error = max(max_error, float(np.abs(probs - expected).max()))
            if condition == 'identity':
                reference_mels, reference_vectors = mels, vectors
            drift = [representation_drift(a, b, x, y, model.scale, config['db_min'], config['db_max'])
                     for a, b, x, y in zip(reference_mels, mels, reference_vectors, vectors, strict=True)]
            logits_delta = ((vectors.astype(np.float64) - reference_vectors) / model.scale) @ model.coefficients.T
            ordered = np.sort(scores)
            track['conditions'][condition] = {
                'predicted_label': labels[int(scores.argmax())], 'scores': scores.tolist(),
                'top_two_probability_gap': float(ordered[-1] - ordered[-2]),
                'identity_class_probability_margin': float(scores[original_class] - np.delete(scores, original_class).max()),
                'mean_window_logit_delta': logits_delta.mean(axis=0).tolist(),
                'windows': drift}
            track_features.append(vectors)
        if sha256_file(audio_path) != row['source_sha256']:
            raise ValueError('Source changed during diagnostic')
        records.append(track)
        all_features.append(np.stack(track_features))
        print(json.dumps({'completed': len(records), 'total': 8, 'max_probability_error': max_error}), flush=True)
    summary = {}
    for condition in conditions:
        windows = [w for r in records for w in r['conditions'][condition]['windows']]
        summary[condition] = {
            'mean_scaled_delta_l2': float(np.mean([w['scaled_delta_l2'] for w in windows])),
            'mean_embedding_relative_l2': float(np.mean([w['embedding_relative_l2'] for w in windows])),
            'mean_cosine_distance': float(np.mean([w['cosine_distance'] for w in windows])),
            'channels': [{field: float(np.mean([w['channels'][i][field] for w in windows]))
                          for field in windows[0]['channels'][i] if field != 'channel'} for i in range(3)]}
    # Descriptive context on ALL 64 previously computed tracks; no extra inference.
    margins = {'changed': [], 'stable': []}
    for r in previous['results']:
        identity = np.array(r['conditions']['identity']['prediction']['scores'])
        noisy = np.array(r['conditions']['white_noise_snr_20_db']['prediction']['scores'])
        group = 'changed' if identity.argmax() != noisy.argmax() else 'stable'
        ordered = np.sort(identity)
        margins[group].append(float(ordered[-1] - ordered[-2]))
    margin_summary = {k: {'tracks': len(v), 'median': float(np.median(v)), 'min': min(v), 'max': max(v)} for k, v in margins.items()}
    if any(sha256_file(ROOT / p) != digest for p, digest in hashes.items()):
        raise ValueError('Source or input changed during diagnostic')
    features_path = ROOT / 'reports/NOISE_DIAGNOSTIC_FEATURES.npz'
    np.savez_compressed(features_path, embeddings=np.stack(all_features))
    result = {'status': 'complete', 'test_evaluated': False, 'test_audio_loaded': False,
              'track_ids': ids, 'conditions': conditions, 'labels': labels, 'budget_seconds': 600,
              'elapsed_seconds': perf_counter() - started, 'model_id': predictor.model_id,
              'source_and_input_sha256': hashes, 'previous_report': pointer,
              'max_window_probability_error': max_error, 'summary': summary,
              'full_panel_original_margins_by_noise_flip': margin_summary,
              'feature_archive': {'path': features_path.relative_to(ROOT).as_posix(), 'sha256': sha256_file(features_path),
                                  'axes': ['track', 'condition', 'window', 'embedding_dimension']},
              'results': records}
    report_path = ROOT / 'reports/NOISE_SENSITIVITY_DIAGNOSTIC.json'
    write_json(report_path, result)
    write_json(ROOT / 'reports/latest_noise_sensitivity_diagnostic.json', {'report': report_path.relative_to(ROOT).as_posix(), 'report_sha256': sha256_file(report_path)})
    print(json.dumps({'summary': summary, 'margins': margin_summary, 'seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
