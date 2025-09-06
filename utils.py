import os
import torch


def load_experiment_results(base_dir, metric_file="epoch_10.pt"):
    """Load experiment results from checkpoint directories."""
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist")
        return []

    results = []

    def get_metric_value(metric_dict, key):
        value = metric_dict.get(key, 0.0)
        return value[-1] if isinstance(value, list) else value

    for dirname in os.listdir(base_dir):
        dirpath = os.path.join(base_dir, dirname)
        if not os.path.isdir(dirpath):
            continue

        try:
            parts = dirname.split("_")
            if "bidir" in dirname or "unidir" in dirname:
                # RNN format: h{hidden_size}_l{num_layers}_{bidir/unidir}_lr{lr}_{optimizer}
                result = {
                    "hidden_size": int(parts[0][1:]),
                    "num_layers": int(parts[1][1:]),
                    "bidirectional": parts[2] == "bidir",
                    "lr": float(parts[3][2:]),
                    "optimizer": parts[4],
                    "direction": (
                        "bidirectional" if parts[2] == "bidir" else "unidirectional"
                    ),
                }
            else:
                # MLP format: h{hidden_dim}_d{dropout}_lr{lr}_{optimizer}
                result = {
                    "hidden_dim": int(parts[0][1:]),
                    "dropout": float(parts[1][1:]),
                    "lr": float(parts[2][2:]),
                    "optimizer": parts[3],
                }
        except (IndexError, ValueError) as e:
            print(f"Skipping {dirname}: {e}")
            continue

        metrics_path = os.path.join(dirpath, metric_file)
        if not os.path.exists(metrics_path):
            print(f"No {metric_file} file found in {dirpath}")
            continue

        try:
            metrics = torch.load(metrics_path, map_location="cpu")
            result.update(
                {
                    "train_acc": get_metric_value(metrics, "train_acc"),
                    "val_acc": get_metric_value(metrics, "val_acc"),
                    "train_loss": get_metric_value(metrics, "train_loss"),
                    "val_loss": get_metric_value(metrics, "val_loss"),
                    "train_f1": get_metric_value(metrics, "train_f1_macro"),
                    "val_f1": get_metric_value(metrics, "val_f1_macro"),
                }
            )
            results.append(result)
            print(
                f"Loaded {dirname}: val_acc={result['val_acc']:.4f}, val_f1={result['val_f1']:.4f}"
            )

        except Exception as e:
            print(f"Error loading metrics from {dirname}: {e}")
            continue

    print(f"\nSuccessfully loaded results from {len(results)} experiments")
    return results
