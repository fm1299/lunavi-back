"""Read the team's test points on the Moon and Mars and compare them with published values:

    uv run python -m pipeline.check_sites                # all ten points, Mars climate included
    uv run python -m pipeline.check_sites --no-climate   # skip the slow Mars Climate Database

The points and their sources are in data/reference/puntos_prueba.json. The report goes to
data/results/site_checks.json. Elevation here is a mean over the site window, while the reference
is a point, so a crater floor reads higher than its landing spot; differences are reported, not
judged. Composition is judged twice: against the rule's expected class (does the app read the map
right?) and against what the lander or rover found (how far does the global map go?).
"""

import argparse
import json
from pathlib import Path

from pipeline.contract import load_config
from pipeline.sites import SiteBook

POINTS = Path("data/reference/puntos_prueba.json")
OUT = Path("data/results/site_checks.json")


def _value(target, key):
    entry = target.features.get(key)
    return None if entry is None else entry.value


def _difference(measured, reference):
    if measured is None or reference is None:
        return None
    return round(float(measured) - float(reference), 1)


def check_point(book: SiteBook, point: dict, climate: bool) -> dict:
    target = book.resolve(point["cuerpo"], point["lat"], point["lon"])
    if climate and target.climate_status == "pending":
        target = book.finish_climate(target.id)
    composition = _value(target, "composition")
    in_situ = point.get("composition_in_situ", point["composition_esperada"])
    return {
        "site": point["sitio"],
        "body": point["cuerpo"],
        "site_id": target.id,
        "climate_status": target.climate_status,
        "elevation_m": _value(target, "elevation"),
        "elevation_reference_m": point["elevation_m"],
        "elevation_difference_m": _difference(_value(target, "elevation"), point["elevation_m"]),
        "temp_diurnal_range_k": _value(target, "temp_diurnal_range"),
        "temp_diurnal_range_reference_k": point["temp_diurnal_range_k"],
        "mean_temperature_c": _value(target, "mean_temperature"),
        "mean_temperature_reference_c": point["mean_temperature_c"],
        "temp_annual_range_k": _value(target, "temp_annual_range"),
        "composition": composition,
        "composition_note": target.features["composition"].note,
        "composition_expected_by_rule": point["composition_esperada"],
        "rule_read_right": composition == point["composition_esperada"],
        "composition_in_situ": in_situ,
        "matches_in_situ": composition == in_situ,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-climate", action="store_true", help="skip the Mars climate query")
    args = parser.parse_args()

    config = load_config()
    book = SiteBook(window_km=config.run.site_window_km, rules=config.rules)
    points = json.loads(POINTS.read_text(encoding="utf-8"))
    rows = []
    for point in points:
        row = check_point(book, point, climate=not args.no_climate)
        rows.append(row)
        mark = "ok" if row["rule_read_right"] else "RULE MISMATCH"
        field = (
            "same as in situ" if row["matches_in_situ"] else f"in situ {row['composition_in_situ']}"
        )
        print(
            f"{row['site'][:48]:48} {str(row['composition']):12} {mark:13} ({field}); "
            f"elevation {row['elevation_m']} vs {row['elevation_reference_m']} m; "
            f"day-night {row['temp_diurnal_range_k']} vs {row['temp_diurnal_range_reference_k']} K",
            flush=True,
        )
    summary = {
        "points": len(rows),
        "rule_read_right": sum(r["rule_read_right"] for r in rows),
        "matches_in_situ": sum(r["matches_in_situ"] for r in rows),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "points": rows}, indent=2) + "\n", "utf-8")
    print(f"{summary}\nWrote {OUT}")


if __name__ == "__main__":
    main()
