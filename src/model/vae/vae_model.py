import torch
import torch.nn as nn



class VAE(nn.Module):

    def __init__(self,input_dim=18000,num_layers=2,hidden_dim=128,latent_dim=32):
        super().__init__()
        
        encoder_layers  = []

        # self.gene_embeddings = nn.Linear(1,latent_dim)
        self.drug_emb = torch.nn.Embedding(146,latent_dim)

        input_size = input_dim
        for i in range(num_layers):
            encoder_layers.append(nn.Linear(input_size,hidden_dim))
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.ReLU())

            input_size = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        self.mu = nn.Linear(hidden_dim,latent_dim)
        self.log_var = nn.Linear(hidden_dim,latent_dim)

        decoder_layers = []
        for i in range(num_layers):
            in_features = latent_dim if i == 0 else hidden_dim
            out_features = input_dim if i == num_layers - 1 else hidden_dim
            decoder_layers.append(nn.Linear(in_features, out_features))
            if i < num_layers - 1:
                decoder_layers.append(nn.BatchNorm1d(out_features))
                decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)


    def _forward_enc(self,x):
        enc = self.encoder(x)

        l_mu = self.mu(enc)
        l_logvar = self.log_var(enc)

        # reparametatrization trick

        sigma = torch.exp(0.5 * l_logvar)
        noise = torch.randn_like(sigma)
        z = l_mu + noise * sigma

        return z, l_mu, l_logvar


    def forward(self,x,drug_emb):
        drug_emb = self.drug_emb(drug_emb)
        z, l_mu, l_logvar = self._forward_enc(x)

        dec = self.decoder(z + drug_emb)
        
        return dec, l_mu,l_logvar


if __name__ == "__main__":
    model = VAE()
    print(model)