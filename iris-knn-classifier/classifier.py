"""Data Classification Using AI -- DecodeLabs AI Project 2.

IPO pipeline on the Iris benchmark:
    INPUT   -> load dataset, split (shuffled + stratified), scale features
    PROCESS -> pick K by cross-validation, train K-Nearest Neighbors
    OUTPUT  -> accuracy, confusion matrix, precision / recall / F1
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.datasets import load_iris  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, train_test_split  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

RANDOM_STATE = 42
TEST_SIZE = 0.20  # the 80/20 split from the slides


# ---------------------------------------------------------------- INPUT
def load_data():
    """Load Iris: 150 samples, 4 features, 3 classes."""
    iris = load_iris()
    return iris.data, iris.target, list(iris.feature_names), list(iris.target_names)


def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Shuffle, then split. Stratify keeps class balance in both sets."""
    return train_test_split(
        X, y, test_size=test_size, shuffle=True, stratify=y, random_state=random_state
    )


# -------------------------------------------------------------- PROCESS
def build_model(k):
    """Scaler + KNN in one pipeline.

    Putting the scaler inside the pipeline means it is fitted on training
    data only, so no test-set information leaks into training.
    """
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))


def sweep_k(X_train, y_train, k_values=range(1, 26), cv=5):
    """Error rate for each K, measured by cross-validation on TRAIN data only."""
    errors = {}
    for k in k_values:
        scores = cross_val_score(build_model(k), X_train, y_train, cv=cv)
        errors[k] = 1.0 - scores.mean()
    return errors


def best_k(errors):
    """Lowest error wins; ties go to the larger (smoother) K."""
    lowest = min(errors.values())
    return max(k for k, e in errors.items() if np.isclose(e, lowest))


# --------------------------------------------------------------- OUTPUT
def evaluate(model, X_test, y_test, class_names):
    preds = model.predict(X_test)
    return {
        "accuracy": float((preds == y_test).mean()),
        "confusion_matrix": confusion_matrix(y_test, preds),
        "report": classification_report(
            y_test, preds, target_names=class_names, digits=3, zero_division=0
        ),
    }


def save_k_curve(errors, chosen_k, path):
    ks = list(errors)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, [errors[k] for k in ks], marker="o")
    ax.axvline(chosen_k, color="tab:red", linestyle="--", label=f"chosen K = {chosen_k}")
    ax.set_xlabel("K value")
    ax.set_ylabel("Cross-validated error rate")
    ax.set_title("Choosing K (training data only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_confusion_matrix(cm, class_names, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, colorbar=False)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(k=None, out_dir="docs"):
    X, y, feature_names, class_names = load_data()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(class_names)} classes")
    print(f"Features: {', '.join(feature_names)}")
    print(f"Classes:  {', '.join(class_names)}\n")

    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Split: {len(X_train)} train / {len(X_test)} test\n")

    errors = sweep_k(X_train, y_train)
    if k is None:
        k = best_k(errors)
        print(f"K selected by 5-fold cross-validation: {k}")
    else:
        print(f"K set manually: {k}")

    model = build_model(k).fit(X_train, y_train)
    results = evaluate(model, X_test, y_test, class_names)

    print(f"\nTest accuracy: {results['accuracy']:.3f}\n")
    print("Confusion matrix (rows = true, cols = predicted):")
    print(results["confusion_matrix"], "\n")
    print(results["report"])

    os.makedirs(out_dir, exist_ok=True)
    save_k_curve(errors, k, os.path.join(out_dir, "k_selection.png"))
    save_confusion_matrix(
        results["confusion_matrix"], class_names, os.path.join(out_dir, "confusion_matrix.png")
    )
    print(f"Plots saved to ./{out_dir}/")
    return results


def main():
    parser = argparse.ArgumentParser(description="Iris classification with KNN")
    parser.add_argument("-k", type=int, default=None, help="neighbors (default: auto)")
    parser.add_argument("--out-dir", default="docs", help="where to save plots")
    args = parser.parse_args()
    run(k=args.k, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
