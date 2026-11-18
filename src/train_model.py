"""Subject-wise cross-validated sleep stage classifier (ML baseline)."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import GroupKFold

from batch_data import load_epoch_table
from constants import SEED, STAGE_NAMES, seed_everything

if __name__ == "__main__":
    seed_everything()
    df = load_epoch_table(sleep_only=True)   # trims the long wake before/after sleep
    feature_cols = [c for c in df.columns if c.endswith(("_logpow", "_relpow"))]
    X, y, subject = df[feature_cols].to_numpy(), df["stage"].to_numpy(), df["subject"].to_numpy()

    # GroupKFold keeps all of a person's epochs (both nights) in the same fold.
    y_pred = np.zeros_like(y)
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups=subject)):
        clf = HistGradientBoostingClassifier(class_weight="balanced", max_iter=200,
                                             random_state=SEED)
        clf.fit(X[tr], y[tr])
        y_pred[te] = clf.predict(X[te])
        print(f"Fold {fold}: balanced acc = {balanced_accuracy_score(y[te], y_pred[te]):.3f}")

    print("\nOverall balanced accuracy:", round(balanced_accuracy_score(y, y_pred), 3))
    print("Cohen's kappa:", round(cohen_kappa_score(y, y_pred), 3))
    print("\nConfusion matrix (rows = true, cols = predicted)")
    print(STAGE_NAMES)
    print(confusion_matrix(y, y_pred))
