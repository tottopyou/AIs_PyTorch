import torch
import torch.nn as nn


def get_tensor_from_the_text (text):

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

    text_tokens = text.split()
    token_to_index = {token: i for i, token in enumerate(text_tokens)}
    indexed_text = [token_to_index[token] for token in text_tokens]

    text_input = torch.tensor(indexed_text).unsqueeze(0)

    tensor = text_encoder(text_input)
    return tensor
