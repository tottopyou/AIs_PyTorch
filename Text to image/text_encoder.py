import torch
import torch.nn as nn

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, text):

        text = text.to(self.embedding.weight.device).long()
        embedded = self.embedding(text)
        _, (hidden, _) = self.rnn(embedded)
        return hidden.squeeze(0)

vocab_size = 10000
embedding_dim = 100
hidden_dim = 256

text_encoder = TextEncoder(vocab_size, embedding_dim, hidden_dim)

torch.save(text_encoder, "pretrained_text_encoder.pth")
