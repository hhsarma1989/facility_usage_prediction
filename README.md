# Facility Usage Prediction System

A reproducible, standard-library Python prototype for predicting the next facility booking from synthetic residential-community history. It creates the dataset, trains/evaluates on a chronological holdout, writes a prediction review CSV, and produces a JSON metrics report.

## Run

Requires Python 3.10+ and no third-party packages.

```bash
python app.py
```

The default command generates `data/bookings.csv`, evaluates the final 20% of bookings in chronological order, and writes `outputs/prediction_review.csv` and `outputs/metrics.json`. To create a fresh deterministic dataset, run `python app.py --generate`. Options include `--rows`, `--seed`, and `--output-dir`.

## Dataset

Each row is a booking with `record_ref`, `resident_ref`, `facility_ref`, `booking_timestamp`, and `usage_timestamp`. The synthetic generator gives residents preferences, facilities different popularity, weekday/hour patterns, booking lead times, sparse activity, occasional noise, and a gradual behavior change. Generated records are sorted by booking time.

## Modeling and leakage controls

The prototype is a transparent, online history model. At each chronological row, it scores facility, usage weekday, usage hour, and booking lead time from that resident's prior rows only. With little personal history it backs off to community-level history; with no history it uses the most common observed values or fixed defaults. Tie breaks are deterministic. Notification timing is reported as a predicted lead duration (for example, "18h before use"). The current row's actual facility, usage time, and booking time are never added to history until after its predictions are made.

This approach was chosen over a more complex classifier because the deliverable emphasizes reproducibility and leakage safety, while the synthetic data intentionally includes sparse residents. Alternatives include per-output gradient-boosted models and collaborative filtering; those need more data and tuning than this prototype warrants.

## Evaluation

The chronological holdout is the latest 20% of booking records. It is not used to fit model parameters or tune thresholds; the model updates its historical state as time advances, using only rows that would already have happened. Metrics include facility accuracy, weekday accuracy, hour accuracy and MAE, notification lead-time MAE, four-output exact-match rate, and per-output match rates. The CSV has each prediction, actual value, match flag, and overall row match count. Since usage day is a weekday rather than an exact date, notification time is compared by lead duration instead of assigning an unsupported calendar date.

## Limitations

This is a synthetic-data prototype, not a calibrated production recommender. The notification-time target is the historical booking timestamp, used as a proxy for when to notify. Actual notification effectiveness, resident availability, cancellations, capacity, holidays, privacy/consent, and facility closures are not modeled. Real deployment would need consented data, operational constraints, calibration, monitoring, and a clearly defined notification objective.

## Files

- `app.py`: generator, leakage-safe predictor, evaluator, and CLI.
- `data/bookings.csv`: deterministic generated data (created on first run).
- `outputs/prediction_review.csv`: holdout predictions side-by-side with actuals.
- `outputs/metrics.json`: overall and per-output performance plus brief error analysis.
