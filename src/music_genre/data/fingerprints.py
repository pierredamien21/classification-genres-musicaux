"""Conservative candidate retrieval; matches are suspicions, not adjudicated duplicates."""

from collections import Counter, defaultdict
from itertools import combinations

import numpy as np


def informative(words, settings):
    return (len(words) >= settings["minimum_words"]
            and len(set(words)) >= settings["minimum_unique_words"])


def compare_fingerprints(a, b, settings):
    positions = defaultdict(list)
    for index, word in enumerate(b):
        positions[word].append(index)
    offsets = Counter(j - i for i, word in enumerate(a) for j in positions[word])
    best = None
    for offset, _ in offsets.most_common(settings["offset_candidates"]):
        start_a, start_b = max(0, -offset), max(0, offset)
        length = min(len(a) - start_a, len(b) - start_b)
        if length < settings["minimum_overlap_words"]:
            continue
        x = np.asarray(a[start_a:start_a + length], dtype=np.uint32)
        y = np.asarray(b[start_b:start_b + length], dtype=np.uint32)
        error = sum(int(v).bit_count() for v in np.bitwise_xor(x, y)) / (32 * length)
        candidate = {"offset_words": offset, "overlap_words": length, "bit_error_rate": error}
        if best is None or error < best["bit_error_rate"]:
            best = candidate
    return best


def find_candidates(records, settings):
    usable = {r["track_id"]: r["fingerprint"] for r in records
              if r["returncode"] == 0 and informative(r["fingerprint"], settings)}
    postings = defaultdict(list)
    for track_id, words in usable.items():
        for word in sorted(set(words)):
            postings[word].append(track_id)
    shared = Counter()
    for tracks in postings.values():
        if 2 <= len(tracks) <= settings["maximum_posting_tracks"]:
            shared.update(combinations(sorted(tracks), 2))
    matches = []
    for (a, b), count in sorted(shared.items()):
        if count < settings["minimum_shared_words"]:
            continue
        score = compare_fingerprints(usable[a], usable[b], settings)
        if score and score["bit_error_rate"] <= settings["maximum_bit_error_rate"]:
            matches.append({"a": a, "b": b, "shared_distinct_words": count, **score})
    return matches
