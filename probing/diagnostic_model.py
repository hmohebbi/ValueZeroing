import torch
import torch.nn as nn


class DiagNet(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.MLP = nn.Sequential(
                        nn.Dropout(0.2),
                        nn.Linear(config.hidden_size, 128),
                        nn.ReLU(),
                        nn.LayerNorm(128),
                        nn.Dropout(0.3),
                        nn.Linear(128, config.num_labels),
                        nn.Softmax(),
                    )

    def forward(self, inputs):
        logits = self.MLP(inputs)
        
        return logits
    