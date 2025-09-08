import torch
from torch.utils.data import DataLoader
from models import Wav2VecFeatureExtractor
from dataset_builder import *
from audio_dataset import *

batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = DataSetBuilder("./data/train/audio/")
ds.map_data()
ds.split_with_lists(
    "./data/train/validation_list.txt",
    "./data/train/testing_list.txt",
)

train_dataset = AudioDataset(ds.train)
val_dataset = AudioDataset(ds.val)
test_dataset = AudioDataset(ds.test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

feature_extractor = Wav2VecFeatureExtractor(freeze=True, pool=False).to(device)
feature_extractor.eval()


def compute_and_save_embeddings(loader, split_name, out_dir="./embeddings_unpooled"):
    all_features, all_labels = [], []
    with torch.no_grad():
        for waveforms, labels in loader:
            waveforms = waveforms.to(device)
            embeddings = feature_extractor(waveforms)  # [B, 768] if pooled
            all_features.append(embeddings.cpu())
            all_labels.append(labels)

    features_tensor = torch.cat(all_features)
    labels_tensor = torch.cat(all_labels)

    out_path = f"{out_dir}/{split_name}_embeddings.pt"
    torch.save({"features": features_tensor, "labels": labels_tensor}, out_path)
    print(f"Saved {split_name} embeddings: {features_tensor.shape} -> {out_path}")


compute_and_save_embeddings(train_loader, "train")
compute_and_save_embeddings(val_loader, "val")
compute_and_save_embeddings(test_loader, "test")
