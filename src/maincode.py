"""Phase 0 pipeline: build/load the shared epoch table and sanity-check the rules.

Run:  python src/maincode.py
Writes small summary CSVs to results/ (safe to commit).
"""
import pandas as pd

from batch_data import load_epoch_table, split_by_subject
from constants import MIN_WAKE_EPOCHS, RESULTS_DIR, STAGE_NAMES, seed_everything


def night_summary(df):
    """One row per subject-night: how long was the sleep period, and how many awakenings?"""
    rows = []
    for (subject, night), g in df.groupby(["subject", "night"]):
        sp = g[g["in_sleep_period"]]
        rows.append({
            "subject": subject,
            "night": night,
            "n_epochs_recorded": len(g),
            "sleep_period_hours": len(sp) * 30 / 3600,
            "n_awakenings": int(sp["awakening"].sum()),
            "has_final_awakening": bool(sp["final_awakening"].any()),
            "start_clock_hour": g["clock_hour"].iloc[0],
        })
    return pd.DataFrame(rows)


def awakening_preceding_stage(df):
    """Which stage was the person in just before each mid-night awakening?"""
    df = df.sort_values(["subject", "night", "epoch_idx"])
    prev = df.groupby(["subject", "night"])["stage"].shift(1)
    events = df[df["awakening"]].assign(prev_stage=prev[df["awakening"]])
    counts = events["prev_stage"].value_counts().sort_index()
    counts.index = [STAGE_NAMES[int(s)] for s in counts.index]
    return counts.rename("n_awakenings").rename_axis("preceding_stage").reset_index()


if __name__ == "__main__":
    seed_everything()
    RESULTS_DIR.mkdir(exist_ok=True)
    print("Starting pipeline...")

    full = load_epoch_table(sleep_only=False)
    sleep = full[full["in_sleep_period"]]
    print(f"\nTable: {len(full):,} epochs total, {len(sleep):,} inside the sleep period")
    print(f"Subjects: {full['subject'].nunique()}, nights: {full.groupby(['subject', 'night']).ngroups}")

    # --- sanity checks on the sleep-period and awakening rules ---
    summary = night_summary(full)
    summary.to_csv(RESULTS_DIR / "night_summary.csv", index=False)
    print(f"\nAwakening rule: wake bout >= {MIN_WAKE_EPOCHS} epochs "
          f"({MIN_WAKE_EPOCHS * 30 / 60:.1f} min)")
    print(f"Awakenings per night: median {summary['n_awakenings'].median():.0f}, "
          f"max {summary['n_awakenings'].max()}, "
          f"nights with none: {(summary['n_awakenings'] == 0).sum()}")
    print(f"Nights with a final awakening: {summary['has_final_awakening'].sum()} of {len(summary)}")
    print(f"Sleep period length (hours): median {summary['sleep_period_hours'].median():.1f}, "
          f"range {summary['sleep_period_hours'].min():.1f}-{summary['sleep_period_hours'].max():.1f}")
    print(f"Recording start clock hour: {summary['start_clock_hour'].min():.1f}-"
          f"{summary['start_clock_hour'].max():.1f}")

    prev = awakening_preceding_stage(full)
    prev.to_csv(RESULTS_DIR / "awakening_preceding_stage.csv", index=False)
    print("\nStage before each mid-night awakening:")
    print(prev.to_string(index=False))

    # --- subject-wise split demo ---
    print("\nSubject-wise 80/20 split of the sleep-period epochs:")
    train_idx, test_idx = split_by_subject(sleep["subject"].to_numpy(), test_size=0.2)
    train, test = sleep.iloc[train_idx], sleep.iloc[test_idx]
    assert set(train["subject"]).isdisjoint(test["subject"])
    print("Check passed: no subject appears in both train and test.")
