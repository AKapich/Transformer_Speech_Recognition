import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class Wav2VecFeatureExtractor(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", freeze=True, pool=True):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(model_name)
        self.pool = pool
        # freezing pre-trained weights
        if freeze:
            for param in self.wav2vec.parameters():
                param.requires_grad = False

    def forward(self, waveforms):  # Tensor of shape [B, T]
        outputs = self.wav2vec(
            waveforms
        ).last_hidden_state  # [B, T', 768], T' is for downsampling
        if self.pool:
            embeddings = outputs.mean(dim=1)  # average over time [B, 768]
            return embeddings  # [B, 768] if pooled, else [B, T', 768]
        return outputs


class MLPClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=12, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class RNNClassifier(nn.Module):
    def __init__(
        self,
        input_size=768,
        hidden_size=128,
        num_layers=1,
        output_size=12,
        bidirectional=False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            nonlinearity="tanh",
        )
        factor = 2 if bidirectional else 1
        self.classifier = nn.Linear(hidden_size * factor, output_size)

    def forward(self, x):
        _, hidden = self.rnn(x)

        if self.bidirectional:
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            last_hidden = hidden[-1]

        logits = self.classifier(last_hidden)
        return logits
