import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size=1024, hidden_size=1024, num_layers=2, vocab_size=78):
        super().__init__()
        layers = []
        in_size = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.LeakyReLU())
            in_size = hidden_size
        
        self.network = nn.Sequential(*layers)
        self.ctc_proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        h = self.network(x)
        return h, self.ctc_proj(h)

class LocationAwareAttention(nn.Module):
    def __init__(self, dec_dim=128, enc_dim=1024, att_dim=256, conv_channels=10, kernel_size=100):
        super().__init__()
        self.W_q = nn.Linear(dec_dim, att_dim, bias=False)
        self.W_k = nn.Linear(enc_dim, att_dim, bias=False)
        self.conv = nn.Conv1d(1, conv_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.W_f = nn.Linear(conv_channels, att_dim, bias=False)
        self.v = nn.Linear(att_dim, 1, bias=True)

    def forward(self, query, keys, prev_align):
        q = self.W_q(query).unsqueeze(1)
        k = self.W_k(keys)
        loc = self.conv(prev_align.unsqueeze(1)).transpose(1, 2)
        if loc.size(1) != k.size(1):
            loc = loc[:, :k.size(1), :]
        energy = self.v(torch.tanh(q + k + self.W_f(loc))).squeeze(-1)
        align = F.softmax(energy, dim=-1)
        return torch.bmm(align.unsqueeze(1), keys).squeeze(1), align

class AttentionalGRUDecoder(nn.Module):
    def __init__(self, vocab_size=78, emb_dim=128, enc_dim=1024, dec_dim=128, att_dim=256, conv_channels=10, kernel_size=100, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = LocationAwareAttention(dec_dim, enc_dim, att_dim, conv_channels, kernel_size)
        self.gru_cell = nn.GRUCell(emb_dim + enc_dim, dec_dim)
        self.output_proj = nn.Linear(dec_dim + enc_dim, dec_dim)
        self.att_loss_proj = nn.Linear(dec_dim, vocab_size)

    def forward(self, targets, encoder_hidden):
        batch_size, seq_len = targets.size(0), targets.size(1)
        hidden = torch.zeros(batch_size, self.gru_cell.hidden_size, device=targets.device)
        prev_align = torch.zeros(batch_size, encoder_hidden.size(1), device=targets.device)
        
        emb = self.dropout(self.embedding(targets))
        outputs, logits = [], []
        
        for t in range(seq_len):
            context, prev_align = self.attention(hidden, encoder_hidden, prev_align)
            hidden = self.gru_cell(torch.cat([emb[:, t, :], context], dim=1), hidden)
            out = self.dropout(self.output_proj(torch.cat([hidden, context], dim=1)))
            outputs.append(out)
            logits.append(self.att_loss_proj(out))
            
        return torch.stack(outputs, dim=1), torch.stack(logits, dim=1)

class PPRM(nn.Module):
    def __init__(self, vocab_size=78, ssl_dim=1024):
        super().__init__()
        self.encoder = Encoder(input_size=ssl_dim, hidden_size=1024, num_layers=2, vocab_size=vocab_size)
        self.decoder = AttentionalGRUDecoder(vocab_size=vocab_size, emb_dim=128, enc_dim=1024, dec_dim=128)

    def forward(self, ssl_features, target_labels):
        enc_h, ctc_logits = self.encoder(ssl_features)
        dec_r, att_logits = self.decoder(target_labels, enc_h)
        return ctc_logits, dec_r, att_logits