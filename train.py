import argparse
import json
import os
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from models import MLPClassifier, RNNClassifier, SpectrogramCNN
from trainer import Trainer


def load_data(data_type, data_dir, batch_size=32, weighted_sampling=False):
    if data_type == "embeddings":
        paths = {
            "train": os.path.join(data_dir, "train_embeddings.pt"),
            "val": os.path.join(data_dir, "val_embeddings.pt"),
            "test": os.path.join(data_dir, "test_embeddings.pt"),
        }

        data = {split: torch.load(path) for split, path in paths.items()}

        train_dataset = TensorDataset(
            data["train"]["features"], data["train"]["labels"]
        )
        val_dataset = TensorDataset(data["val"]["features"], data["val"]["labels"])
        test_dataset = TensorDataset(data["test"]["features"], data["test"]["labels"])

    elif data_type == "spectrograms":
        paths = {
            "train": os.path.join(data_dir, "train_spectrograms.pt"),
            "val": os.path.join(data_dir, "val_spectrograms.pt"),
            "test": os.path.join(data_dir, "test_spectrograms.pt"),
        }

        data = {
            split: torch.load(path, map_location="cpu") for split, path in paths.items()
        }

        for split in data:
            if len(data[split]["features"].shape) == 3:
                data[split]["features"] = data[split]["features"].unsqueeze(1)

        train_dataset = TensorDataset(
            data["train"]["features"], data["train"]["labels"]
        )
        val_dataset = TensorDataset(data["val"]["features"], data["val"]["labels"])
        test_dataset = TensorDataset(data["test"]["features"], data["test"]["labels"])

    else:
        raise ValueError(
            f"Unknown data_type: {data_type}. Use 'embeddings' or 'spectrograms'"
        )

    if weighted_sampling:
        train_labels = data["train"]["labels"]
        counts = torch.bincount(train_labels)
        weights = 1.0 / counts.float()
        weights = weights / weights.sum() * len(counts)
        sample_weights = weights[train_labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset


def create_model(model_type, config, device):
    models = {
        "mlp": MLPClassifier(
            input_dim=config.get("input_size", 768),
            hidden_dim=config.get("hidden_dim", 256),
            num_classes=config.get("num_classes", 12),
            dropout=config.get("dropout", 0.3),
        ),
        "rnn": RNNClassifier(
            input_size=config.get("input_size", 768),
            hidden_size=config.get("hidden_size", 128),
            num_layers=config.get("num_layers", 1),
            output_size=config.get("num_classes", 12),
            bidirectional=config.get("bidirectional", False),
        ),
        "cnn": SpectrogramCNN(
            input_channels=config.get("input_channels", 1),
            conv_channels=config.get("conv_channels", [16, 32, 64]),
            kernel_sizes=config.get("kernel_sizes", [3, 3, 3]),
            pool_sizes=config.get("pool_sizes", [2, 2, 2]),
            fc_hidden_dim=config.get("fc_hidden_dim", 128),
            num_classes=config.get("num_classes", 12),
            dropout=config.get("dropout", 0.3),
        ),
    }

    if model_type == "cnn_staging":
        return create_two_stage_cnn(config, device)

    if model_type not in models:
        raise ValueError(f"Unknown model_type: {model_type}")

    return models[model_type].to(device)


def create_two_stage_cnn(config, device):
    base_config = {
        "input_channels": config.get("input_channels", 1),
        "conv_channels": config.get("conv_channels", [16, 32, 64]),
        "kernel_sizes": config.get("kernel_sizes", [3, 3, 3]),
        "pool_sizes": config.get("pool_sizes", [2, 2, 2]),
        "fc_hidden_dim": config.get("fc_hidden_dim", 128),
        "dropout": config.get("dropout", 0.3),
    }

    stage1_model = SpectrogramCNN(**base_config, num_classes=2).to(device)
    stage2_model = SpectrogramCNN(
        **base_config, num_classes=config.get("num_known_classes", 11)
    ).to(device)

    return stage1_model, stage2_model


def prepare_staging_data(train_dataset, val_dataset, unknown_class_index=11):
    train_features = train_dataset.tensors[0]
    train_labels = train_dataset.tensors[1]
    val_features = val_dataset.tensors[0]
    val_labels = val_dataset.tensors[1]

    binary_train_labels = (train_labels == unknown_class_index).long()
    binary_val_labels = (val_labels == unknown_class_index).long()

    stage1_train_dataset = TensorDataset(train_features, binary_train_labels)
    stage1_val_dataset = TensorDataset(val_features, binary_val_labels)

    known_mask = train_labels != unknown_class_index
    stage2_train_dataset = TensorDataset(
        train_features[known_mask], train_labels[known_mask]
    )

    known_val_mask = val_labels != unknown_class_index
    stage2_val_dataset = TensorDataset(
        val_features[known_val_mask], val_labels[known_val_mask]
    )

    return (
        stage1_train_dataset,
        stage1_val_dataset,
        stage2_train_dataset,
        stage2_val_dataset,
    )


def train_model(
    model_type, config, train_loader, val_loader, checkpoint_dir, epochs, device
):
    model = create_model(model_type, config, device)

    if config.get("weighted_loss", False):
        train_labels = []
        for _, labels in train_loader:
            train_labels.extend(labels.numpy())
        train_labels = torch.tensor(train_labels)

        counts = torch.bincount(train_labels)
        weights = 1.0 / counts.float()
        weights = weights / weights.sum() * len(counts)
        criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    else:
        criterion = torch.nn.CrossEntropyLoss()

    trainer = Trainer(
        classifier=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_name=config.get("optimizer", "adam"),
        lr=config.get("lr", 1e-3),
        criterion=criterion,
    )

    print(f"Training {model_type} model for {epochs} epochs...")
    trainer.fit(epochs=epochs, checkpoint_dir=checkpoint_dir)

    return model, trainer


def train_two_stage_cnn(
    config, train_dataset, val_dataset, checkpoint_dir, epochs, device, batch_size=16
):
    print("Starting Two-Stage CNN Training...")

    stage1_train, stage1_val, stage2_train, stage2_val = prepare_staging_data(
        train_dataset, val_dataset, config.get("unknown_class_index", 11)
    )

    stage1_model, stage2_model = create_two_stage_cnn(config, device)

    def create_weighted_loader(dataset, batch_size):
        labels = dataset.tensors[1]
        counts = torch.bincount(labels)
        weights = 1.0 / counts.float()
        weights = weights / weights.sum() * len(counts)
        sample_weights = weights[labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler), weights

    def train_stage(stage_name, model, train_data, val_data, weights):
        print(f"\n=== {stage_name} ===")

        train_loader, class_weights = create_weighted_loader(train_data, batch_size)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

        trainer = Trainer(
            classifier=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer_name=config.get("optimizer", "adam"),
            lr=config.get("lr", 1e-3),
            criterion=criterion,
        )

        stage_dir = os.path.join(
            checkpoint_dir, stage_name.lower().replace(" ", "_").split(":")[1].strip()
        )
        trainer.fit(epochs=epochs, checkpoint_dir=stage_dir)
        return trainer

    stage1_trainer = train_stage(
        "STAGE 1: Binary Classification (Known vs Unknown)",
        stage1_model,
        stage1_train,
        stage1_val,
        None,
    )
    stage2_trainer = train_stage(
        "STAGE 2: Multi-class Classification (Known Classes Only)",
        stage2_model,
        stage2_train,
        stage2_val,
        None,
    )

    return stage1_model, stage2_model, stage1_trainer, stage2_trainer


def main():
    parser = argparse.ArgumentParser(description="Train speech recognition models")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["mlp", "rnn", "cnn", "cnn_staging"],
        help="Model type to train",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        required=True,
        choices=["embeddings", "spectrograms"],
        help="Type of input data",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the data files",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--weighted_sampling",
        action="store_true",
        help="Use weighted sampling for imbalanced datasets",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training",
    )

    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    print(f"Using device: {device}")

    with open(args.config, "r") as f:
        config = json.load(f)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(os.path.join(args.checkpoint_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Loading {args.data_type} data from {args.data_dir}...")
    train_loader, val_loader, test_loader, train_dataset = load_data(
        args.data_type, args.data_dir, args.batch_size, args.weighted_sampling
    )

    if args.model in ["mlp", "rnn"] and args.data_type != "embeddings":
        raise ValueError(f"Model {args.model} requires embeddings data")
    if args.model in ["cnn", "cnn_staging"] and args.data_type != "spectrograms":
        raise ValueError(f"Model {args.model} requires spectrograms data")

    if args.model == "cnn_staging":
        val_dataset = TensorDataset(
            val_loader.dataset.tensors[0], val_loader.dataset.tensors[1]
        )
        train_two_stage_cnn(
            config,
            train_dataset,
            val_dataset,
            args.checkpoint_dir,
            args.epochs,
            device,
            args.batch_size,
        )
        print("Two-stage CNN training completed!")
    else:
        train_model(
            args.model,
            config,
            train_loader,
            val_loader,
            args.checkpoint_dir,
            args.epochs,
            device,
        )
        print(f"{args.model.upper()} training completed!")

    print(f"Checkpoints saved to: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
