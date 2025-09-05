import torch
import torchaudio
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, data_mapping, sample_rate=16000, load_audio=True):
        self.items = list(data_mapping.items())
        self.sample_rate = sample_rate
        self.max_length = 1 * sample_rate
        self.load_audio = load_audio

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        file_path, label = self.items[idx]

        if not self.load_audio:
            return file_path, label

        waveform, sr = torchaudio.load(file_path)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if waveform.size(1) > self.max_length:
            waveform = waveform[:, : self.max_length]
        else:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.max_length - waveform.size(1))
            )

        return waveform.squeeze(0), label
