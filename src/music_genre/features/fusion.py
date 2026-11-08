"""Join frozen representations only when their audio windows match exactly."""

import numpy as np


def concatenate_features(vit, descriptors, vit_index, descriptor_index,
                         vit_names, descriptor_names, descriptor_families,
                         excluded_family=None):
    if excluded_family not in (None, "rhythm", "harmony"):
        raise ValueError("Unsupported excluded family")
    if not vit_index or vit_index != descriptor_index:
        raise ValueError("Representations must have identical indexed windows")
    seen = set()
    for position, row in enumerate(vit_index):
        if row["split"] not in ("train", "validation"):
            raise ValueError("Only training and validation windows are allowed")
        key = (row["track_id"], int(row["window_id"]))
        if key in seen or int(row["row_index"]) != position:
            raise ValueError("Duplicate or unordered window index")
        seen.add(key)
    if (vit.shape != (len(vit_index), len(vit_names))
            or descriptors.shape != (len(descriptor_index), len(descriptor_names))
            or len(descriptor_names) != len(descriptor_families)
            or not np.isfinite(vit).all() or not np.isfinite(descriptors).all()):
        raise ValueError("Invalid feature matrices or metadata")
    if set(descriptor_families) != {"timbre", "harmony", "rhythm"}:
        raise ValueError("Unexpected descriptor families")
    selected = [i for i, family in enumerate(descriptor_families) if family != excluded_family]
    names = list(vit_names) + [f"descriptor/{descriptor_names[i]}" for i in selected]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate feature names")
    families = ["vit"] * len(vit_names) + [descriptor_families[i] for i in selected]
    return np.concatenate((vit, descriptors[:, selected]), axis=1), names, families, selected
