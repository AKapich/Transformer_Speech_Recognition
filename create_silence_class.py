import os
import librosa
import soundfile as sf

background_dir = r"./data/train/audio/_background_noise_"
silence_dir = os.path.join(os.path.dirname(background_dir), "silence")
os.makedirs(silence_dir, exist_ok=True)


for filename in os.listdir(background_dir):
    if filename.endswith(".wav"):
        file_path = os.path.join(background_dir, filename)

        audio, sr = librosa.load(file_path, sr=None)
        samples_per_second = sr
        total_seconds = len(audio) // samples_per_second

        for i in range(total_seconds):
            start_sample = i * samples_per_second
            end_sample = (i + 1) * samples_per_second
            clip = audio[start_sample:end_sample]

            base_name = os.path.splitext(filename)[0]
            clip_name = f"silence_{base_name}_{i+1}.wav"
            clip_path = os.path.join(silence_dir, clip_name)
            sf.write(clip_path, clip, sr)
