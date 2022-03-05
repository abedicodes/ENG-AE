import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, n_features, hidden_dim, num_layers):
        super(Encoder, self).__init__()

        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn1 = nn.LSTM(input_size=self.n_features, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.rnn2 = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True)

    def forward(self, x):

        x, (h_n, c_n) = self.rnn1(x)
        x, (h_n, c_n) = self.rnn2(x)
        return h_n.squeeze()


class Decoder(nn.Module):
    def __init__(self, n_features, hidden_dim, num_layers):
        super(Decoder, self).__init__()

        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn1 = nn.LSTM(input_size=self.n_features, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.rnn2 = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True)

        self.output_layer = nn.Linear(self.hidden_dim, hidden_dim).cuda()

    def forward(self, x, seq_len):

        x = x.repeat(seq_len, 1, 1)
        x = x.permute(1, 0, 2)

        x, (h_n, c_n) = self.rnn1(x)
        x, (h_n, c_n) = self.rnn2(x)

        x = self.output_layer(x)

        return x


class lstmae(nn.Module):
    def __init__(self, n_features, hidden_dim, num_layers):
        super(lstmae, self).__init__()

        self.encoder = Encoder(n_features, hidden_dim, num_layers)
        self.decoder = Decoder(hidden_dim, n_features, num_layers)

    def forward(self, x):
        seq_len = x.shape[1]
        x = self.encoder(x)
        x = self.decoder(x, seq_len)
        return x
