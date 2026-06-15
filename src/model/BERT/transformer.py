import torch
import torch.nn as nn
from attetntion import Attention


class Transformer(nn.Module):
    def __init__(self, attn_layers=2, embeding_dim=512, num_drugs=146, seq_len=18211):
        super().__init__()
        self.attn = nn.ModuleList([Attention(embeding_dim=embeding_dim) for _ in range(attn_layers)])
        self.lm_head = nn.Linear(embeding_dim, 1)
        self.control_embedding = nn.Linear(1, embeding_dim)
        self.drug_name_embedding = nn.Embedding(num_drugs, embeding_dim)
        self.seq_len = seq_len

    def forward(self, control_de, drug_name):
        control_de = control_de.unsqueeze(-1)
        control_de = self.control_embedding(control_de)
        drug = self.drug_name_embedding(drug_name).unsqueeze(1)
        x = control_de + drug
        for layer in self.attn:
            x = layer(x)
        return self.lm_head(x).squeeze(-1)
