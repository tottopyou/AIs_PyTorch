import torch
import torch.nn as nn


def get_tensor_from_the_text (text):
    class TextEncoder(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim):
            super(TextEncoder, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        def forward(self, text):

            text = text.to(self.embedding.weight.device).long()  # Move to the same device as embedding
            embedded = self.embedding(text)
            _, (hidden, _) = self.rnn(embedded)
            return hidden.squeeze(0)


            # Example usage
    vocab_size = 10000  # Example vocab size
    embedding_dim = 100
    hidden_dim = 256

    # Initialize the text encoder
    text_encoder = TextEncoder(vocab_size, embedding_dim, hidden_dim)


    # Tokenization
    text_tokens = text.split()  # Split text into tokens
    token_to_index = {token: i for i, token in enumerate(text_tokens)}
    indexed_text = [token_to_index[token] for token in text_tokens]  # Convert tokens to indices

    # Convert to PyTorch tensor
    text_input = torch.tensor(indexed_text).unsqueeze(0)  # Add batch dimension

    # Forward pass
    tensor = text_encoder(text_input)
    print(tensor)
    return tensor
