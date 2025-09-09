# Transformer Speech Recognition

This project implements and compares two main approaches for speech recognition:

1. **Wav2Vec2 Embeddings Approach**: pre-trained Wav2Vec2 transformers for extracting audio features, followed by simple classifiers (MLP/RNN)
2. **CNN Spectrogram Approach**: Audio converted to mel spectrograms and passed to convolutional neural network architecture

## How to train the models?

The `train.py` script provides a unified interface for training all model types. Here's how to use it:

### Basic Usage

```bash
python train.py --model <MODEL_TYPE> --data_type <DATA_TYPE> --data_dir <DATA_DIRECTORY> --config <CONFIG_FILE> --checkpoint_dir <CHECKPOINT_DIRECTORY>
```

### Model Types Available

- `mlp`: Multi-Layer Perceptron (requires embeddings)
- `rnn`: Recurrent Neural Network (requires embeddings) 
- `cnn`: Convolutional Neural Network (requires spectrograms)
- `cnn_staging`: Two-stage CNN system (requires spectrograms)

### Example Commands

**Train MLP with embeddings:**
```bash
python train.py --model mlp --data_type embeddings --data_dir embeddings --config configs/mlp_config.json --checkpoint_dir checkpoints/MLP_experiment --epochs 40
```

**Train CNN with spectrograms:**
```bash
python train.py --model cnn --data_type spectrograms --data_dir spectrograms --config configs/cnn_config.json --checkpoint_dir checkpoints/CNN_experiment --epochs 60
```

**Train RNN with custom batch size:**
```bash
python train.py --model rnn --data_type embeddings --data_dir embeddings --config configs/rnn_config.json --checkpoint_dir checkpoints/RNN_experiment --epochs 50 --batch_size 64
```

### Configuration Files

Create JSON configuration files in the `configs/` directory. Example for MLP:

```json
{
    "input_size": 768,
    "hidden_dim": 256,
    "num_classes": 12,
    "dropout": 0.3,
    "optimizer": "adam",
    "lr": 0.001,
    "weighted_loss": true
}
```

### Command Line Options

- `--model`: Model architecture (mlp/rnn/cnn/cnn_staging)
- `--data_type`: Input data type (embeddings/spectrograms)
- `--data_dir`: Directory containing preprocessed data files
- `--config`: Path to JSON configuration file
- `--checkpoint_dir`: Directory to save model checkpoints
- `--epochs`: Number of training epochs (default: 40)
- `--batch_size`: Training batch size (default: 32)
- `--weighted_sampling`: Enable weighted sampling for imbalanced datasets
- `--device`: Training device (auto/cpu/cuda, default: auto)

### Prerequisites

Prepare your dataset by downloading audio files to the `data/` directory and running the appropriate preprocessing script - `precompute_embeddings.py` for MLP/RNN models or `precompute_spectrograms.py` for CNN models.



