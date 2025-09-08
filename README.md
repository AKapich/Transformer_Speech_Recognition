# Transformer Speech Recognition

This project implements and compares two main approaches for speech recognition:

1. **Wav2Vec2 Embeddings Approach**: Uses pre-trained Wav2Vec2 transformers to extract audio features, followed by simple classifiers (MLP/RNN)
2. **CNN Spectrogram Approach**: Converts audio to mel spectrograms and uses Convolutional Neural Networks for classification
