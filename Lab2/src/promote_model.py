"""
Champion/challenger model promotion.

models/registry.json records the current best model (the "champion") and every
model that was ever promoted. A newly trained model (the "challenger") only
replaces the champion if its F1 score is strictly higher.

Usage:
    python src/promote_model.py check   --version <v> [--force]
    python src/promote_model.py promote --version <v>

"check" prints the decision and, inside GitHub Actions, writes promote=true/false
to $GITHUB_OUTPUT and a comparison table to $GITHUB_STEP_SUMMARY.
"promote" copies the model and metrics into models/ and metrics/ and updates the registry.

Only uses the standard library, so the promote job does not need to install anything.
"""

import argparse
import json
import os
import shutil
import sys

REGISTRY_PATH = os.path.join("models", "registry.json")
METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc")


def load_registry(path=REGISTRY_PATH):
    if not os.path.exists(path):
        return {"champion": None, "history": []}
    with open(path) as f:
        return json.load(f)


def should_promote(challenger_f1, registry, force=False):
    """
    Returns:
        tuple: (bool decision, str reason)
    """
    champion = registry.get("champion")
    if force:
        return True, "Forced promotion requested."
    if champion is None:
        return True, "No champion yet, so the first model that passes the gate is promoted."
    if challenger_f1 > champion["metrics"]["f1"]:
        return True, f"Challenger F1 {challenger_f1:.4f} beats champion F1 {champion['metrics']['f1']:.4f}."
    return False, f"Challenger F1 {challenger_f1:.4f} does not beat champion F1 {champion['metrics']['f1']:.4f}."


def read_build(version, build_dir):
    with open(os.path.join(build_dir, f"metrics_{version}.json")) as f:
        metrics = json.load(f)
    with open(os.path.join(build_dir, f"params_{version}.json")) as f:
        params = json.load(f)
    return metrics, params


def summary_table(metrics, params, champion):
    """
    Builds a Markdown table comparing the challenger with the current champion.
    """
    champ_metrics = champion["metrics"] if champion else {}
    champ_label = f"Champion `{champion['version']}`" if champion else "Champion (none)"
    lines = [
        f"| Metric | Challenger `{params['version']}` | {champ_label} |",
        "|---|---|---|",
        f"| model | {params['model_type']} | {champion['params']['model_type'] if champion else '-'} |",
    ]
    for name in METRIC_NAMES:
        champ_value = f"{champ_metrics[name]:.4f}" if name in champ_metrics else "-"
        lines.append(f"| {name} | {metrics[name]:.4f} | {champ_value} |")
    return "\n".join(lines)


def check(version, build_dir, force):
    metrics, params = read_build(version, build_dir)
    registry = load_registry()
    decision, reason = should_promote(metrics["f1"], registry, force)

    table = summary_table(metrics, params, registry.get("champion"))
    verdict = "PROMOTE" if decision else "KEEP CHAMPION"
    print(table)
    print(f"\nDecision: {verdict}. {reason}")

    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"promote={'true' if decision else 'false'}\n")
    if "GITHUB_STEP_SUMMARY" in os.environ:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(f"## Model evaluation\n\n{table}\n\n**Decision: {verdict}.** {reason}\n")
    return decision


def promote(version, build_dir):
    metrics, params = read_build(version, build_dir)
    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    model_path = os.path.join("models", f"model_{version}.joblib")
    shutil.copy(os.path.join(build_dir, f"model_{version}.joblib"), model_path)
    with open(os.path.join("metrics", f"metrics_{version}.json"), "w") as f:
        json.dump({"version": version, "params": params, "metrics": metrics}, f, indent=4)

    registry = load_registry()
    entry = {"version": version, "model_path": model_path, "params": params, "metrics": metrics}
    registry["champion"] = entry
    registry["history"].append(entry)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=4)

    print(f"Promoted {version} to champion ({model_path})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check", "promote"))
    parser.add_argument("--version", required=True)
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--force", default="false", help="'true' to promote regardless of the champion")
    args = parser.parse_args()

    if args.command == "check":
        check(args.version, args.build_dir, args.force.lower() == "true")
    else:
        promote(args.version, args.build_dir)


if __name__ == "__main__":
    sys.exit(main())
