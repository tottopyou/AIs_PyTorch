import torch
import torch.nn as nn

# Define and initialize the text encoder
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, text):

        text = text.to(self.embedding.weight.device).long()  # Move to the same device as embedding
        embedded = self.embedding(text)
        _, (hidden, _) = self.rnn(embedded)
        return hidden.squeeze(0)  # Squeeze to remove the sequence length dimension

# Example usage
vocab_size = 10000  # Example vocab size
embedding_dim = 100
hidden_dim = 256

# Initialize the text encoder
text_encoder = TextEncoder(vocab_size, embedding_dim, hidden_dim)

# Save the text encoder to a file
torch.save(text_encoder, "pretrained_text_encoder.pth")
