import torch
import torch.nn as nn
from tqdm import tqdm
import os
from sklearn.metrics import f1_score


class Trainer:
    def __init__(
        self,
        classifier,
        train_loader,
        val_loader=None,
        device=None,
        optimizer_name="adam",
        lr=1e-3,
        dropout=0.2,
        criterion=None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if hasattr(classifier, "net"):
            for layer in classifier.net:
                if isinstance(layer, nn.Dropout):
                    layer.p = dropout

        self.classifier = classifier.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = criterion or nn.CrossEntropyLoss()

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
        all_preds, all_labels = [], []

        for features, labels in tqdm(self.train_loader, desc="Training"):
            features, labels = features.to(self.device), labels.to(self.device)

            outputs = self.classifier(features)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        avg_loss = total_loss / len(all_labels)
        accuracy = (all_preds == all_labels).sum().item() / len(all_labels)
        macro_f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average="macro")

        return avg_loss, accuracy, macro_f1

    @torch.no_grad()
    def evaluate(self, loader=None):
        loader = loader or self.val_loader
        if loader is None:
            raise ValueError("No evaluation loader provided")

        self.classifier.eval()
        total_loss = 0
        all_preds, all_labels = [], []

        for features, labels in tqdm(loader, desc="Evaluating"):
            features, labels = features.to(self.device), labels.to(self.device)

            outputs = self.classifier(features)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        avg_loss = total_loss / len(all_labels)
        accuracy = (all_preds == all_labels).sum().item() / len(all_labels)
        macro_f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average="macro")

        return avg_loss, accuracy, macro_f1

    def fit(self, epochs, checkpoint_dir="./checkpoints", start_epoch=0):
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir

        for epoch in range(1 + start_epoch, epochs + 1 + start_epoch):
            train_loss, train_acc, train_f1 = self.train_epoch()
            print(
                f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1 (macro): {train_f1:.4f}"
            )

            if self.val_loader is not None:
                val_loss, val_acc, val_f1 = self.evaluate(self.val_loader)
                print(
                    f"           Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1 (macro): {val_f1:.4f}"
                )
            else:
                val_loss = val_acc = val_f1 = None

            self.save_checkpoint(
                epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1
            )

    def save_checkpoint(
        self,
        epoch,
        train_loss,
        train_acc,
        train_f1,
        val_loss,
        val_acc,
        val_f1,
    ):
        path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "classifier_state": self.classifier.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_f1_macro": train_f1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1_macro": val_f1,
            },
            path,
        )
        print(f"Checkpoint saved: {path}")
