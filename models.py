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
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        _, hidden = self.rnn(x)

        if self.bidirectional:
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            last_hidden = hidden[-1]

        logits = self.classifier(last_hidden)
        return logits


class SpectrogramCNN(nn.Module):
    def __init__(
        self,
        input_channels=1,
        conv_channels=[16, 32, 64],  # number of filters per conv layer
        kernel_sizes=[3, 3, 3],
        pool_sizes=[2, 2, 2],
        fc_hidden_dim=128,
        num_classes=12,
        dropout=0.3,
    ):
        super().__init__()
        assert len(conv_channels) == len(kernel_sizes) == len(pool_sizes)

        self.conv_layers = nn.ModuleList()
        in_ch = input_channels
        for out_ch, k, p in zip(conv_channels, kernel_sizes, pool_sizes):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=k // 2),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=p),
                )
            )
            in_ch = out_ch

        self.dropout = nn.Dropout(dropout)
        self.fc_hidden_dim = fc_hidden_dim
        self.num_classes = num_classes
        self._fc_input_dim = None

        self.fc1 = None
        self.fc2 = nn.Linear(fc_hidden_dim, num_classes)

    def _get_conv_output_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            for conv_layer in self.conv_layers:
                dummy_input = conv_layer(dummy_input)
            return dummy_input.shape[
                1
            ]  # after global average pooling, the size will be the number of channels

    def forward(self, x):
        if self.fc1 is None:
            conv_output_size = self._get_conv_output_size(x.shape[1:])
            self.fc1 = nn.Linear(conv_output_size, self.fc_hidden_dim).to(x.device)

        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = x.mean(dim=[2, 3])  # Shape: [batch_size, channels]

        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
