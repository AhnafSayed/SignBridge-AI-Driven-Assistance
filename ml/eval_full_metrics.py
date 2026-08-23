"""Full classification metrics for the shipped .tflite models, in the form
an academic report needs: accuracy, macro/weighted precision-recall-F1,
top-5 accuracy, per-category breakdown, and the worst/best classes.

Why this exists separately from eval_confidence.py: that script answers an
*engineering* question (where should the app's commit threshold sit?).
This one answers the *reporting* question (what are this model's real
precision/recall/F1, per category, on held-out data?) -- until now the
project only ever reported accuracy, which is a single number that hides
how differently the word/digit/letter portions actually behave.

Both scripts deliberately reuse eval_confidence's loaders so the train/val
split, the cache keys, and the GroupShuffleSplit seed are identical to the
ones the models were actually trained with -- these numbers are computed
on genuinely held-out data, not re-scored training data.

Usage:
    python eval_full_metrics.py --language asl --data-dir /path/to/asl-signs
    python eval_full_metrics.py --language bdsl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from eval_confidence import category_of, load_asl, load_bdsl, run_tflite  # noqa: E402


def top_k_accuracy(scores: np.ndarray, y_true: np.ndarray, k: int = 5) -> float:
    """Fraction of samples whose true label is among the model's k highest
    scoring outputs. Reported because for a 250-way task, "the right answer
    was in the model's top handful of guesses" is a meaningful, standard
    measure alongside strict top-1 -- and it's the number most directly
    comparable to the isolated-sign-recognition literature.
    """
    if scores.shape[1] <= k:
        return 1.0
    topk = np.argpartition(-scores, k, axis=1)[:, :k]
    return float(np.mean([y_true[i] in topk[i] for i in range(len(y_true))]))


def summarize(label_names, y_true, scores, language):
    preds = scores.argmax(axis=1)
    categories = np.array([category_of(label_names[i]) for i in y_true])

    report = {"language": language, "n_val": int(len(y_true)), "n_classes": len(label_names)}

    report["overall"] = {
        "accuracy": float(np.mean(preds == y_true)),
        "top5_accuracy": top_k_accuracy(scores, y_true, 5),
        "chance": 1.0 / len(label_names),
    }
    p, r, f, _ = precision_recall_fscore_support(y_true, preds, average="macro", zero_division=0)
    report["overall"].update({"macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f)})
    p, r, f, _ = precision_recall_fscore_support(y_true, preds, average="weighted", zero_division=0)
    report["overall"].update(
        {"weighted_precision": float(p), "weighted_recall": float(r), "weighted_f1": float(f)}
    )

    # Per-category: the headline number for this project is meaningless
    # without this split, because words (hard, held-out-signer, many
    # classes) and digits/letters (easier, random split, few classes) sit
    # in the same model and average together into one flattering figure.
    report["by_category"] = {}
    for cat in ("word", "digit", "letter"):
        mask = categories == cat
        if not mask.any():
            continue
        cat_true, cat_pred = y_true[mask], preds[mask]
        p, r, f, _ = precision_recall_fscore_support(
            cat_true, cat_pred, average="macro", zero_division=0, labels=np.unique(cat_true)
        )
        # In-app accuracy: the app's mode switch restricts the argmax to
        # the active category (SignCategory / setAllowedCategories), so a
        # digit signed in Digits mode can only ever be scored against the
        # other nine digits -- it can never lose to a word. The unmasked
        # number above is the honest *model* metric; this is the honest
        # *product* metric, and for digits the two differ a lot because
        # their errors are overwhelmingly cross-category (digit macro
        # precision is 1.0 -- when the model does say a digit it is never
        # wrong, it just gets outscored by word classes). Reporting only
        # one of these would mislead in one direction or the other.
        cat_label_idx = np.array(
            [i for i, name in enumerate(label_names) if category_of(name) == cat]
        )
        masked_pred = cat_label_idx[scores[mask][:, cat_label_idx].argmax(axis=1)]

        report["by_category"][cat] = {
            "n": int(mask.sum()),
            "n_classes": int(len(np.unique(cat_true))),
            "accuracy": float(np.mean(cat_pred == cat_true)),
            "accuracy_in_app_masked": float(np.mean(masked_pred == cat_true)),
            "top5_accuracy": top_k_accuracy(scores[mask], cat_true, 5),
            "macro_precision": float(p),
            "macro_recall": float(r),
            "macro_f1": float(f),
        }

    # Best/worst classes -- the honest detail an aggregate hides.
    per_class = {}
    for idx in np.unique(y_true):
        m = y_true == idx
        per_class[label_names[idx]] = {"n": int(m.sum()), "recall": float(np.mean(preds[m] == idx))}
    ranked = sorted(per_class.items(), key=lambda kv: kv[1]["recall"])
    report["worst_10_classes"] = [{"label": k, **v} for k, v in ranked[:10]]
    report["best_10_classes"] = [{"label": k, **v} for k, v in ranked[-10:][::-1]]

    # Per-class recall *as the app scores it* -- argmax restricted to the
    # class's own category. This is the list that matters for deciding what
    # to tell the user: a class with zero unmasked recall may be perfectly
    # recognizable in its own mode (it was merely outscored by a word),
    # whereas a class that is still at zero *here* genuinely cannot be
    # recognized and should be flagged in the interface rather than offered
    # as if it worked.
    masked_recall = {}
    for cat in ("word", "digit", "letter"):
        cat_label_idx = np.array(
            [i for i, name in enumerate(label_names) if category_of(name) == cat]
        )
        if not len(cat_label_idx):
            continue
        m = categories == cat
        if not m.any():
            continue
        mp = cat_label_idx[scores[m][:, cat_label_idx].argmax(axis=1)]
        mt = y_true[m]
        for idx in np.unique(mt):
            sel = mt == idx
            masked_recall[label_names[idx]] = {
                "n": int(sel.sum()),
                "recall_in_app": float(np.mean(mp[sel] == idx)),
                "category": cat,
            }
    ranked_masked = sorted(masked_recall.items(), key=lambda kv: kv[1]["recall_in_app"])
    report["worst_15_classes_in_app"] = [{"label": k, **v} for k, v in ranked_masked[:15]]
    report["unrecognizable_in_app"] = [
        k for k, v in masked_recall.items() if v["recall_in_app"] == 0.0
    ]
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--language", choices=["asl", "bdsl"], required=True)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--alphabet-path", default=None)
    p.add_argument("--digit-path", default=None)
    p.add_argument("--digits-dir", default=None)
    p.add_argument("--letters-dir", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--out", default=None, help="Write the report as JSON here too")
    args = p.parse_args()

    if args.language == "asl":
        X, y, signers, label_names = load_asl(args)
        default_model = os.path.join(
            os.path.dirname(__file__), "..", "app", "assets", "models", "asl", "model.tflite"
        )
    else:
        X, y, signers, label_names = load_bdsl(args)
        default_model = os.path.join(
            os.path.dirname(__file__), "..", "app", "assets", "models", "bdsl", "model.tflite"
        )
    model_path = args.model or default_model

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, val_idx = next(splitter.split(X, y, groups=signers))
    X_val, y_val = X[val_idx], y[val_idx]
    print(f"Val: {len(val_idx)} samples, {len(set(signers[val_idx]))} unseen groups, {len(label_names)} classes")

    scores = run_tflite(model_path, X_val)
    report = summarize(label_names, y_val, scores, args.language)

    o = report["overall"]
    print(f"\n=== {args.language.upper()} | {report['n_classes']} classes | n={report['n_val']} ===")
    print(f"accuracy        {o['accuracy']:.4f}   (chance {o['chance']:.4f})")
    print(f"top-5 accuracy  {o['top5_accuracy']:.4f}")
    print(f"macro    P/R/F1 {o['macro_precision']:.4f} / {o['macro_recall']:.4f} / {o['macro_f1']:.4f}")
    print(f"weighted P/R/F1 {o['weighted_precision']:.4f} / {o['weighted_recall']:.4f} / {o['weighted_f1']:.4f}")
    print("\ncategory   n     classes  acc     inApp   top5    macroP  macroR  macroF1")
    for cat, m in report["by_category"].items():
        print(
            f"{cat:9s} {m['n']:6d} {m['n_classes']:6d}  {m['accuracy']:.4f}  "
            f"{m['accuracy_in_app_masked']:.4f}  {m['top5_accuracy']:.4f}  "
            f"{m['macro_precision']:.4f}  {m['macro_recall']:.4f}  {m['macro_f1']:.4f}"
        )
    print(f"  acc = unmasked {report['n_classes']}-way argmax (model metric)")
    print("  inApp = argmax restricted to the active category, as the mode switch does (product metric)")
    print("\nWorst classes as the app scores them (argmax within the active mode):")
    for c in report["worst_15_classes_in_app"]:
        print(f"  {c['label']:24s} n={c['n']:4d}  recall_in_app={c['recall_in_app']:.3f}")
    unrec = report["unrecognizable_in_app"]
    print(f"\nUnrecognizable even within their own mode ({len(unrec)}): {', '.join(unrec) or 'none'}")

    print("\nWorst classes (unmasked recall):")
    for c in report["worst_10_classes"]:
        print(f"  {c['label']:24s} n={c['n']:4d}  recall={c['recall']:.3f}")

    out = args.out or os.path.join(os.path.dirname(__file__), f"metrics_{args.language}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
