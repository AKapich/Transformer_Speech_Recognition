import os
import torch
import torchaudio
from torch.utils.data import TensorDataset
from tqdm import tqdm


class SpectrogramProcessor:
    def __init__(
        self,
        sample_rate=16000,
        n_mels=128,
        n_fft=2048,
        hop_length=256,
        max_time_frames=64,
        device=None,
        apply_normalization=True,
        apply_delta_features=False,
    ):
        self.sample_rate = sample_rate
        self.max_time_frames = max_time_frames
        self.apply_normalization = apply_normalization
        self.apply_delta_features = apply_delta_features
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Main spectrogram transform
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=80,
            f_max=8000,
        ).to(self.device)

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(self.device)

        if self.apply_delta_features:
            self.delta_transform = torchaudio.transforms.ComputeDeltas().to(self.device)

    def process_file(self, input_wav_path):
        waveform, original_sr = torchaudio.load(input_wav_path)
        if original_sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, original_sr, self.sample_rate
            )
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # mel spectrogram
        mel_spec = self.mel_spectrogram(waveform.to(self.device))
        log_mel_spec = self.amplitude_to_db(mel_spec)
        # normalization
        if self.apply_normalization:
            log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (
                log_mel_spec.std() + 1e-8
            )

        # consistent time dimension
        current_time_frames = log_mel_spec.size(-1)
        if current_time_frames > self.max_time_frames:
            log_mel_spec = log_mel_spec[..., : self.max_time_frames]
        elif current_time_frames < self.max_time_frames:
            pad_amount = self.max_time_frames - current_time_frames
            log_mel_spec = torch.nn.functional.pad(log_mel_spec, (0, pad_amount))

        if self.apply_delta_features:
            delta_features = self.delta_transform(log_mel_spec)
            log_mel_spec = torch.cat([log_mel_spec, delta_features], dim=0)

        return log_mel_spec.cpu()

    def process_split(self, file_list, labels, out_path):
        all_specs, all_labels = [], []
        for path, label in tqdm(
            zip(file_list, labels), total=len(file_list), desc=f"Processing {out_path}"
        ):
            spec = self.process_file(path)
            all_specs.append(spec)
            all_labels.append(torch.tensor(label))

        features = torch.stack(all_specs)
        labels_tensor = torch.stack(all_labels)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save({"features": features, "labels": labels_tensor}, out_path)
        print(f"Saved spectrograms -> {out_path}")
        return TensorDataset(features, labels_tensor)
