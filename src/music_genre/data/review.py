"""Deterministic, conservative selection; preserves an explicit decision for every track."""

from collections import defaultdict


def review_rows(rows, fingerprints, candidates, settings):
    by_id = {r["track_id"]: r for r in rows}
    if len(by_id) != len(rows) or set(fingerprints) != set(by_id):
        raise ValueError("Audit and fingerprint IDs must be unique and identical")
    parents = {key: key for key in by_id}

    def root(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def join(a, b):
        ra, rb = root(a), root(b)
        parents[max(ra, rb)] = min(ra, rb)

    exact = defaultdict(list)
    for row in rows:
        if row["pcm_sha256"]:
            exact[row["pcm_sha256"]].append(row["track_id"])
    for group in exact.values():
        for other in group[1:]:
            join(group[0], other)
    for pair in candidates:
        join(pair["a"], pair["b"])
    groups = defaultdict(list)
    for key in by_id:
        groups[root(key)].append(key)
    uncertain_components = {root(pair["a"]) for pair in candidates if not pair["exact_pcm"]}
    result = {}
    benign_flags = {"near_full_scale", "duplicate_pcm_sha256", "duplicate_source_sha256"}
    for track_id, row in by_id.items():
        flags = set(filter(None, row["flags"].split("|")))
        record = fingerprints[track_id]
        reasons = sorted(flags - benign_flags)
        if record["returncode"] != 0:
            reasons.append("fingerprint_extraction_failed")
        if record["diagnostic"]:
            reasons.append("independent_decoder_diagnostic")
        if (len(record["fingerprint"]) < settings["minimum_words"]
                or len(set(record["fingerprint"])) < settings["minimum_unique_words"]):
            reasons.append("low_information_fingerprint")
        decision = "hold" if reasons else "keep"
        if row["decode_status"] != "ok" or row["status"] == "excluded":
            decision = "exclude"
            reasons.append(row["exclusion_reason"] or "unusable_audio")
        result[track_id] = {
            **row, "audit_status": row["status"], "decision": decision,
            "decision_reasons": reasons, "representative_track_id": "",
            "content_group": root(track_id),
        }
    for group_id, ids in sorted(groups.items()):
        if len(ids) < 2:
            continue
        reasons = []
        if len({by_id[key]["genre"] for key in ids}) > 1:
            reasons.append("content_group_label_conflict")
        if len({by_id[key]["split"] for key in ids}) > 1:
            reasons.append("content_group_cross_split")
        if group_id in uncertain_components:
            reasons.append("nonexact_fingerprint_candidate")
        if reasons:
            for key in ids:
                if result[key]["decision"] != "exclude":
                    result[key]["decision"] = "hold"
                result[key]["decision_reasons"].extend(reasons)
        else:
            eligible = sorted(key for key in ids if result[key]["decision"] == "keep")
            if eligible:
                representative = eligible[0]
                for key in eligible:
                    result[key]["representative_track_id"] = representative
                    if key != representative:
                        result[key]["decision"] = "exclude"
                        result[key]["decision_reasons"].append("redundant_exact_pcm")
    for row in result.values():
        row["decision_reasons"] = "|".join(sorted(set(row["decision_reasons"]))) or (
            "retained_level_flag" if "near_full_scale" in row["flags"] else "technical_checks_passed"
        )
        row["status"] = {"keep": "valid", "hold": "needs_review", "exclude": "excluded"}[row["decision"]]
        row["exclusion_reason"] = row["decision_reasons"] if row["decision"] == "exclude" else ""
    return [result[key] for key in sorted(result)]
