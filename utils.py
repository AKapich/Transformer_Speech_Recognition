import os
import torch
import pandas as pd
from torch.utils.data import TensorDataset

class_names = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "silence",
    "unknown",
]


def load_embeddings(split_name, emb_dir="./embeddings"):
    path = os.path.join(emb_dir, f"{split_name}.pt")
    data = torch.load(path)
    features = data["features"]
    labels = data["labels"]
    return TensorDataset(features, labels)


def load_spectrograms(split_name, spec_dir="./spectrograms"):
    """Load spectrograms from saved files"""
    path = os.path.join(spec_dir, f"{split_name}.pt")
    data = torch.load(path, map_location="cpu")
    spectrograms = data["features"]
    labels = data["labels"]

    if len(spectrograms.shape) == 3:
        spectrograms = spectrograms.unsqueeze(1)

    print(f"Loaded {split_name}: {spectrograms.shape}, Labels: {labels.shape}")
    return TensorDataset(spectrograms, labels)


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
                }
            elif "conv" in dirname:
                # CNN format: conv{conv_str}_fc{fc_hidden_dim}_d{dropout}_lr{lr}_{optimizer}
                result = {
                    "conv_layers": parts[0][4:] + " " + parts[1] + " " + parts[2],
                    "fc_hidden_dim": int(parts[-4][2:]),
                    "dropout": float(parts[-3][1:]),
                    "lr": float(parts[-2][2:]),
                    "optimizer": parts[-1],
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


def load_pt_files_with_epochs(folder_path, target_fields):
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder '{folder_path}' does not exist")
    if not os.path.isdir(folder_path):
        raise ValueError(f"'{folder_path}' is not a directory")

    pt_files = [
        f
        for f in os.listdir(folder_path)
        if f.endswith(".pt") and os.path.isfile(os.path.join(folder_path, f))
    ]
    if not pt_files:
        raise ValueError(f"No .pt files found in '{folder_path}'")

    results = []
    for filename in pt_files:
        try:
            file_path = os.path.join(folder_path, filename)
            data = torch.load(file_path)
            extracted_data = {}
            if isinstance(data, dict):
                for field in target_fields:
                    if field in data:
                        extracted_data[field] = data[field]
                extracted_data["model_params"] = folder_path.split("\\")[-1]
                results.append(extracted_data)
            else:
                print(f"Warning: Could not extract epoch number from '{filename}'")

        except Exception as e:
            print(f"Error processing file '{filename}': {e}")
    results.sort(key=lambda x: x.get("epoch", 0))

    return pd.DataFrame(results)
