import os
import torch
import torchaudio


class SpectrogramProcessor:
    def __init__(self, sample_rate=16000, n_mels=64, n_fft=1024, hop_length=512):
        self.sample_rate = sample_rate
        self.spectrogram_transform = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
            ),
            torchaudio.transforms.AmplitudeToDB(),
        )

    def convert_and_save(self, input_wav_path, output_path):
        waveform, original_sr = torchaudio.load(input_wav_path)

        if original_sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, original_sr, self.sample_rate
            )

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        log_mel_spec = self.spectrogram_transform(waveform)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(log_mel_spec, output_path)
