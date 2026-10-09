#!/usr/bin/env python3
"""Facility usage prediction prototype; Python standard library only."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

FACILITIES = ["Gym", "Pool", "Badminton", "Clubhouse", "Multipurpose Hall"]
HOURS = [6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21]
START = datetime(2025, 1, 1)
MONDAY = datetime(2024, 1, 1)
FIELDS = ["record_ref", "resident_ref", "facility_ref", "booking_timestamp", "usage_timestamp"]


def weighted_choice(rng: random.Random, choices: list, weights: list[float]):
    return rng.choices(choices, weights=weights, k=1)[0]


def generate_dataset(path: Path, rows: int = 3200, seed: int = 17) -> None:
    rng = random.Random(seed)
    residents = [f"R-{i:03d}" for i in range(1, 141)]
    weights = [1.0 / (1 + i * 0.07) for i in range(len(residents))]
    preferences = {}
    for resident in residents:
        preferences[resident] = {
            "fac": weighted_choice(rng, range(len(FACILITIES)), [4, 2, 2, 1, 1]),
            "day": rng.randrange(7),
            "hour": rng.choice(HOURS),
            "lead": rng.choice([2, 8, 18, 30, 48, 72]),
        }
    events = []
    for n in range(rows):
        resident = weighted_choice(rng, residents, weights)
        pref = preferences[resident]
        # A gradual shift in preferences introduces realistic behavior drift.
        drift = n > rows * 0.62 and rng.random() < 0.22
        fac_idx = weighted_choice(rng, range(len(FACILITIES)), [4, 3, 2, 1.5, 1]) if drift else pref["fac"]
        if rng.random() < 0.18:
            fac_idx = weighted_choice(rng, range(len(FACILITIES)), [5, 3, 2, 1, 1])
        day = (6 if drift else pref["day"]) if rng.random() < 0.72 else rng.randrange(7)
        hour = (18 if drift else pref["hour"]) if rng.random() < 0.70 else rng.choice(HOURS)
        lead = pref["lead"] if rng.random() < 0.78 else rng.choice([1, 4, 12, 24, 60, 96])
        # Choose the next requested weekday from a randomly selected week.
        usage_day = START + timedelta(days=rng.randrange(52) * 7)
        usage_day += timedelta(days=(day - usage_day.weekday()) % 7)
        usage = usage_day.replace(hour=hour, minute=rng.choice([0, 0, 15, 30, 45]))
        booking = usage - timedelta(hours=lead)
        events.append((booking, usage, resident, FACILITIES[fac_idx]))
    events.sort(key=lambda x: (x[0], x[1], x[2]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for i, (booked, usage, resident, facility) in enumerate(events, 1):
            writer.writerow({"record_ref": f"B-{i:05d}", "resident_ref": resident,
                             "facility_ref": facility, "booking_timestamp": booked.isoformat(timespec="minutes"),
                             "usage_timestamp": usage.isoformat(timespec="minutes")})


def mode(counter: Counter, fallback):
    return min(counter, key=lambda x: (-counter[x], str(x))) if counter else fallback


def predict(hist: list[dict], fallback: dict) -> dict:
    """Make a prediction using history strictly before the scored record."""
    recent = hist[-12:]
    facilities = Counter(r["facility_ref"] for r in recent)
    days = Counter(datetime.fromisoformat(r["usage_timestamp"]).weekday() for r in recent)
    hours = Counter(datetime.fromisoformat(r["usage_timestamp"]).hour for r in recent)
    leads = [int((datetime.fromisoformat(r["usage_timestamp"]) - datetime.fromisoformat(r["booking_timestamp"])).total_seconds() // 3600) for r in recent]
    # Personal history first; community distribution supplies cold-start predictions.
    return {"facility": mode(facilities, fallback["facility"]),
            "day": mode(days, fallback["day"]), "hour": mode(hours, fallback["hour"]),
            "lead": sorted(leads)[len(leads)//2] if leads else fallback["lead"]}


def run(dataset: Path, output_dir: Path) -> dict:
    with dataset.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (r["booking_timestamp"], r["record_ref"]))
    if len(rows) < 10:
        raise ValueError("Dataset must contain at least 10 bookings")
    split = max(1, int(len(rows) * 0.8))
    # Community fallback during holdout is seeded with training rows only.
    train = rows[:split]
    actual_fac = Counter(r["facility_ref"] for r in train)
    actual_day = Counter(datetime.fromisoformat(r["usage_timestamp"]).weekday() for r in train)
    actual_hour = Counter(datetime.fromisoformat(r["usage_timestamp"]).hour for r in train)
    train_leads = sorted(int((datetime.fromisoformat(r["usage_timestamp"]) - datetime.fromisoformat(r["booking_timestamp"])).total_seconds() // 3600) for r in train)
    fallback = {"facility": mode(actual_fac, "Gym"), "day": mode(actual_day, 5),
                "hour": mode(actual_hour, 18), "lead": train_leads[len(train_leads)//2] if train_leads else 24}
    history: dict[str, list[dict]] = defaultdict(list)
    for row in train:
        history[row["resident_ref"]].append(row)
    review = []
    for row in rows[split:]:
        pred = predict(history[row["resident_ref"]], fallback)
        actual_usage = datetime.fromisoformat(row["usage_timestamp"])
        actual_book = datetime.fromisoformat(row["booking_timestamp"])
        day_ok = pred["day"] == actual_usage.weekday()
        hour_ok = pred["hour"] == actual_usage.hour
        fac_ok = pred["facility"] == row["facility_ref"]
        # Notification timing is reported as a lead duration; no exact predicted date
        # is invented because the target is a weekday, not a calendar date.
        lead_actual = (actual_usage - actual_book).total_seconds() / 3600
        lead_error = abs(pred["lead"] - lead_actual)
        nudge_ok = lead_error <= 24
        matches = sum([fac_ok, day_ok, hour_ok, nudge_ok])
        review.append({"record_ref": row["record_ref"], "resident_ref": row["resident_ref"],
                       "predicted_facility": pred["facility"], "actual_facility": row["facility_ref"], "facility_match": fac_ok,
                       "predicted_day": (MONDAY + timedelta(days=pred["day"])).strftime("%a"), "actual_day": actual_usage.strftime("%a"), "day_match": day_ok,
                       "predicted_hour": f"{pred['hour']:02d}:00", "actual_hour": actual_usage.strftime("%H:%M"), "hour_match": hour_ok,
                       "predicted_notification_timing": f"{pred['lead']}h before use",
                       "actual_notification_time": actual_book.isoformat(timespec="minutes"), "notification_within_24h": nudge_ok,
                       "predicted_lead_hours": pred["lead"], "actual_lead_hours": round(lead_actual, 1),
                       "notification_lead_error_hours": round(lead_error, 1), "outputs_matched": f"{matches} of 4",
                       "all_outputs_match": matches == 4})
        history[row["resident_ref"]].append(row)
    n = len(review)
    def rate(key): return sum(bool(x[key]) for x in review) / n
    metrics = {"dataset_rows": len(rows), "training_rows": split, "holdout_rows": n,
               "holdout_start_booking_timestamp": rows[split]["booking_timestamp"],
               "facility_accuracy": round(rate("facility_match"), 4), "weekday_accuracy": round(rate("day_match"), 4),
               "hour_accuracy": round(rate("hour_match"), 4),
               "hour_mae_hours": round(sum(abs(int(x["predicted_hour"][:2]) - int(x["actual_hour"][:2])) for x in review) / n, 2),
               "notification_lead_mae_hours": round(sum(x["notification_lead_error_hours"] for x in review) / n, 2),
               "notification_within_24h_rate": round(rate("notification_within_24h"), 4),
               "four_output_exact_match_rate": round(rate("all_outputs_match"), 4),
               "per_output_match_rates": {"facility": round(rate("facility_match"), 4), "weekday": round(rate("day_match"), 4),
                                          "hour": round(rate("hour_match"), 4), "notification_within_24h": round(rate("notification_within_24h"), 4)},
               "error_analysis": "Notification-time errors are assessed by predicted versus actual booking lead time (within 24 hours); the model predicts weekday rather than an exact usage date."}
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "prediction_review.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(review[0])); writer.writeheader(); writer.writerows(review)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=3200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--generate", action="store_true", help="regenerate deterministic synthetic dataset")
    args = parser.parse_args()
    root = args.output_dir
    data = root / "data" / "bookings.csv"
    if args.generate or not data.exists():
        generate_dataset(data, args.rows, args.seed)
        print(f"Generated {args.rows} bookings: {data}")
    metrics = run(data, root / "outputs")
    print(json.dumps(metrics, indent=2))
    print(f"Review: {root / 'outputs' / 'prediction_review.csv'}")


if __name__ == "__main__":
    main()
