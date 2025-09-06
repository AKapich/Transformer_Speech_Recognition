import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from dataset_builder import *
from audio_dataset import *
from spectrogram_processor import SpectrogramProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = DataSetBuilder("./data/train/audio/")
ds.map_data()
ds.split_with_lists(
    "./data/train/validation_list.txt",
    "./data/train/testing_list.txt",
)

train_dataset = AudioDataset(ds.train, load_audio=False)
val_dataset = AudioDataset(ds.val, load_audio=False)
test_dataset = AudioDataset(ds.test, load_audio=False)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
processor = SpectrogramProcessor(
    sample_rate=16000,
    n_mels=64,
    n_fft=512,
    hop_length=512,
    max_time_frames=32,
    apply_normalization=True,
    apply_delta_features=False,
)


def compute_and_save_spectrograms(
    loader, split_name, processor, out_dir="./spectrograms"
):
    all_specs, all_labels = [], []

    for file_paths, labels in tqdm(
        loader, desc=f"Processing {split_name} spectrograms"
    ):
        batch_specs = []
        for file_path in file_paths:
            spec = processor.process_file(file_path)
            batch_specs.append(spec)

        specs_tensor = torch.stack(batch_specs)
        all_specs.append(specs_tensor)
        all_labels.append(labels)

    features = torch.cat(all_specs)
    labels_tensor = torch.cat(all_labels)

    out_path = f"{out_dir}/{split_name}_spectrograms.pt"
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"features": features, "labels": labels_tensor}, out_path)
    print(f"Saved {split_name} spectrograms: {features.shape} -> {out_path}")

    return TensorDataset(features, labels_tensor)


val_tensor_dataset = compute_and_save_spectrograms(val_loader, "val", processor)
train_tensor_dataset = compute_and_save_spectrograms(train_loader, "train", processor)
test_tensor_dataset = compute_and_save_spectrograms(test_loader, "test", processor)
