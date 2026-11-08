"""Prespecified central-window subset, with no track silently dropped."""


def central_window_rows(index):
    if not index:
        raise ValueError("Empty window index")
    groups = {}
    for position, row in enumerate(index):
        if int(row["row_index"]) != position or row["split"] not in ("train", "validation"):
            raise ValueError("Invalid row order or forbidden split")
        groups.setdefault(row["track_id"], []).append((position, row))
    selected = []
    for windows in groups.values():
        starts = [float(row["start_seconds"]) for _, row in windows]
        if len(windows) != 3 or sorted(starts) != [0., 10., 20.]:
            raise ValueError("Each track must have the three reference windows")
        if len({int(row["window_id"]) for _, row in windows}) != 3:
            raise ValueError("Duplicate window identifier")
        for key in ("split", "label_index", "artist_id", "source_sha256"):
            if len({row[key] for _, row in windows}) != 1:
                raise ValueError("Inconsistent track metadata")
        selected.extend(position for position, row in windows if float(row["start_seconds"]) == 10.)
    selected.sort()
    return selected, [{**index[position], "row_index": str(i)} for i, position in enumerate(selected)]
