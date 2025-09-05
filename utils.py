import os
import torch


def load_mlp_experiment_results(base_dir, metric_file="epoch_10.pt"):
    results = []

    for dirname in os.listdir(base_dir):
        dirpath = os.path.join(base_dir, dirname)
        if not os.path.isdir(dirpath):
            continue

        # Format: h{hidden_dim}_d{dropout}_lr{lr}_{optimizer}
        try:
            parts = dirname.split("_")
            hidden_dim = int(parts[0][1:])
            dropout = float(parts[1][1:])
            lr = float(parts[2][2:])
            optimizer = parts[3]
        except Exception as e:
            print(f"Skipping {dirname}: {e}")
            continue

        metrics_path = os.path.join(dirpath, metric_file)
        if not os.path.exists(metrics_path):
            print(f"No {metric_file} file found in {dirpath}")
            continue

        try:
            metrics = torch.load(metrics_path, map_location="cpu")

            def get_metric_value(metric_dict, key):
                value = metric_dict[key]
                if isinstance(value, list):
                    return value[-1]
                else:
                    return value

            train_f1_key = "train_f1_macro"
            val_f1_key = "val_f1_macro"
            result = {
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "lr": lr,
                "optimizer": optimizer,
                "train_acc": get_metric_value(metrics, "train_acc"),
                "val_acc": get_metric_value(metrics, "val_acc"),
                "train_loss": get_metric_value(metrics, "train_loss"),
                "val_loss": get_metric_value(metrics, "val_loss"),
                "train_f1": (
                    get_metric_value(metrics, train_f1_key) if train_f1_key else 0.0
                ),
                "val_f1": get_metric_value(metrics, val_f1_key) if val_f1_key else 0.0,
            }
            results.append(result)
            print(
                f"✓ Loaded {dirname}: val_acc={result['val_acc']:.4f}, val_f1={result['val_f1']:.4f}"
            )

        except KeyError as e:
            print(f"Missing key in metrics from {dirname}: {e}")
            print(f"Available keys: {list(metrics.keys())}")
            continue
        except Exception as e:
            print(f"Error loading metrics from {dirname}: {e}")
            try:
                print(f"Metrics type: {type(metrics)}")
                if isinstance(metrics, dict):
                    print(f"Available keys: {list(metrics.keys())}")
                    for key, value in list(metrics.items())[:3]:  # first 3 items
                        print(
                            f"  {key}: {type(value)} - {value if not isinstance(value, list) else f'list of length {len(value)}'}"
                        )
            except:
                pass
            continue

    print(f"\nSuccessfully loaded results from {len(results)} experiments")
    return results
