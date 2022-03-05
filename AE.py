from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_dim=68, embedding_dim=64):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        self.fc = nn.Linear(self.input_dim, self.embedding_dim * 2)
        self.fc_ = nn.Linear(self.embedding_dim * 2, self.embedding_dim)

    def forward(self, x):
        return self.fc_(self.fc(x))


class Decoder(nn.Module):
    def __init__(self, input_dim=68, embedding_dim=64):
        super(Decoder, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        self.fc = nn.Linear(self.embedding_dim, self.input_dim * 2)
        self.fc_ = nn.Linear(self.input_dim * 2, self.input_dim)

    def forward(self, x):
        return self.fc_(self.fc(x))


class ae(nn.Module):

    def __init__(self, input_dim=11, embedding_dim=64):
        super(ae, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        self.encoder = Encoder(input_dim=self.input_dim, embedding_dim=self.embedding_dim).cuda()
        self.decoder = Decoder(input_dim=self.input_dim, embedding_dim=self.embedding_dim).cuda()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class bc(nn.Module):
    def __init__(self, input_dim=37, embedding_dim=64):
        super(bc, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        self.fc = nn.Linear(self.input_dim, self.embedding_dim)
        self.fc_ = nn.Linear(self.embedding_dim, 1)

    def forward(self, x):
        return self.fc_(self.fc(x))
