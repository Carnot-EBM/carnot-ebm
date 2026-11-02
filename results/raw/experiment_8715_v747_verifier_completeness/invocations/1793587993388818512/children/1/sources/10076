class ARModelLSTM(nn.Module):
    def __init__(self, vocab_size=6, d_model=64, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, causal=True): # causal arg ignored, it is inherently causal
        e = self.embed(x)
        out, _ = self.lstm(e)
        return self.fc_out(out)
