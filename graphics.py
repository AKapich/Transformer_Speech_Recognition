import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import torch


def plot_loss(df, save_path=None):
    if isinstance(df, list):
        df_plot = pd.DataFrame(df)
    else:
        df_plot = df

    plt.figure(figsize=(9, 6))
    plt.plot(
        df_plot["epoch"],
        df_plot["train_loss"],
        label="Training Loss",
        marker="o",
        linewidth=2.5,
        color="#3cae2c",
        markersize=6,
        alpha=0.8,
    )
    plt.plot(
        df_plot["epoch"],
        df_plot["val_loss"],
        label="Validation Loss",
        marker="p",
        linewidth=2.5,
        color="#d702ae",
        markersize=6,
        alpha=0.8,
    )

    plt.xlabel("Epoch", fontsize=14, fontweight="bold")
    plt.ylabel("Loss", fontsize=14, fontweight="bold")
    plt.title("Training vs Validation Loss", fontsize=16, fontweight="bold", pad=20)

    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    plt.gca().set_facecolor("#f8f9fa")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_linewidth(1.5)
    plt.gca().spines["bottom"].set_linewidth(1.5)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Plot saved to: {save_path}")

    plt.show()


def plot_f1_macro(df, save_path=None):
    """Plot training and validation F1 macro scores over epochs"""
    if isinstance(df, list):
        df_plot = pd.DataFrame(df)
    else:
        df_plot = df

    plt.figure(figsize=(9, 6))
    plt.plot(
        df_plot["epoch"],
        df_plot["train_f1_macro"],
        label="Training F1 Macro",
        marker="o",
        linewidth=2.5,
        color="#3cae2c",
        markersize=6,
        alpha=0.8,
    )
    plt.plot(
        df_plot["epoch"],
        df_plot["val_f1_macro"],
        label="Validation F1 Macro",
        marker="s",
        linewidth=2.5,
        color="#d702ae",
        markersize=6,
        alpha=0.8,
    )

    plt.xlabel("Epoch", fontsize=14, fontweight="bold")
    plt.ylabel("F1 Macro Score", fontsize=14, fontweight="bold")
    plt.title(
        "Training vs Validation F1 Macro Score", fontsize=16, fontweight="bold", pad=20
    )

    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    plt.gca().set_facecolor("#f8f9fa")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_linewidth(1.5)
    plt.gca().spines["bottom"].set_linewidth(1.5)
    plt.ylim(0, 1.05)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Plot saved to: {save_path}")

    plt.show()


def plot_per_class_f1(
    f1_scores, class_names=None, title="Per-Class F1 Scores", save_path=None
):
    colors = plt.cm.RdYlGn(np.array(f1_scores))

    plt.figure(figsize=(12, 8))
    bars = plt.bar(
        range(len(f1_scores)),
        f1_scores,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )

    for _, (bar, score) in enumerate(zip(bars, f1_scores)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    plt.xlabel("Class", fontsize=14, fontweight="bold")
    plt.ylabel("F1 Score", fontsize=14, fontweight="bold")
    plt.title(title, fontsize=16, fontweight="bold", pad=20)

    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.ylim(0, 1.05)

    macro_avg = np.mean(f1_scores)
    plt.axhline(
        y=macro_avg,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"Macro Average: {macro_avg:.3f}",
    )

    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle="--", axis="y")
    plt.tight_layout()

    plt.gca().set_facecolor("#f8f9fa")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_linewidth(1.5)
    plt.gca().spines["bottom"].set_linewidth(1.5)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Plot saved to: {save_path}")

    plt.show()


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names=None,
    title="Confusion Matrix",
    normalize=False,
    save_path=None,
):
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cbar_label = "Normalized Frequency"
    else:
        cbar_label = "Count"

    plt.figure(figsize=(10, 8))
    im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title, fontsize=16, fontweight="bold", pad=20)

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=12, fontweight="bold")
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right", fontsize=10)
    plt.yticks(tick_marks, class_names, fontsize=10)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                text = f"{cm[i, j]:.2f}"
                if cm[i, j] > 0:
                    plt.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        fontweight="bold",
                        fontsize=9,
                        color="white" if cm[i, j] > thresh else "black",
                    )
            else:
                text = f"{cm[i, j]:d}"
                if cm[i, j] > 0:
                    plt.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        fontweight="bold",
                        fontsize=9,
                        color="white" if cm[i, j] > thresh else "black",
                    )

    plt.xlabel("Predicted Label", fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=14, fontweight="bold")
    plt.tight_layout()

    plt.gca().set_facecolor("#f8f9fa")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Plot saved to: {save_path}")

    plt.show()
    return cm


def evaluate_and_plot_full_analysis(
    model,
    test_loader,
    class_names=None,
    device="cpu",
    save_f1_path=None,
    save_cm_path=None,
):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    f1_per_class = f1_score(all_targets, all_preds, average=None)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    print(classification_report(all_targets, all_preds, target_names=class_names))

    plot_per_class_f1(
        f1_per_class,
        class_names,
        title=f"Per-Class F1 Scores (Macro F1: {macro_f1:.3f})",
        save_path=save_f1_path,
    )

    plot_confusion_matrix(
        all_targets,
        all_preds,
        class_names,
        title=f"Confusion Matrix (Accuracy: {np.mean(np.array(all_targets) == np.array(all_preds)):.3f})",
        save_path=save_cm_path,
    )

    return f1_per_class, macro_f1, all_targets, all_preds
