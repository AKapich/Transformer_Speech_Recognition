import torch
import torch.nn as nn
from tqdm import tqdm
import os


class Trainer:
    def __init__(
        self,
        feature_extractor,
        classifier,
        train_loader,
        val_loader=None,
        device=None,
        optimizer_name="adam",
        lr=1e-3,
        dropout=0.2,
        pool=True,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if hasattr(feature_extractor, "pool"):
            feature_extractor.pool = pool
        # Set dropout if supported (MLPClassifier)
        if hasattr(classifier, "net"):
            for layer in classifier.net:
                if isinstance(layer, nn.Dropout):
                    layer.p = dropout
        self.feature_extractor = feature_extractor.to(self.device)
        self.classifier = classifier.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.CrossEntropyLoss()

        params = list(self.classifier.parameters())
        optimizer_map = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
        }

        optimizer_class = optimizer_map.get(optimizer_name.lower())
        if optimizer_class is None:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        self.optimizer = optimizer_class(params, lr=lr)

    def train_epoch(self):
        self.classifier.train()
        total_loss = 0
        correct = 0
        total = 0

        for waveforms, labels in tqdm(self.train_loader, desc="Training"):
            waveforms = waveforms.to(self.device)
            labels = labels.to(self.device)

            # Extract embeddings
            with torch.no_grad():
                embeddings = self.feature_extractor(waveforms)

            outputs = self.classifier(embeddings)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, loader=None):
        loader = loader or self.val_loader
        if loader is None:
            raise ValueError("No evaluation loader provided")

        self.classifier.eval()
        total_loss = 0
        correct = 0
        total = 0

        for waveforms, labels in tqdm(loader, desc="Evaluating"):
            waveforms = waveforms.to(self.device)
            labels = labels.to(self.device)

            embeddings = self.feature_extractor(waveforms)
            outputs = self.classifier(embeddings)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def fit(self, epochs, checkpoint_dir="checkpoints"):
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch()
            print(
                f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
            )

            if self.val_loader is not None:
                val_loss, val_acc = self.evaluate(self.val_loader)
                print(f"           Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch, best=False):
        path = os.path.join(
            self.checkpoint_dir, f"{'best' if best else f'epoch_{epoch}'}.pt"
        )
        torch.save(
            {
                "epoch": epoch,
                "classifier_state": self.classifier.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            },
            path,
        )
        print(f"Checkpoint saved: {path}")
