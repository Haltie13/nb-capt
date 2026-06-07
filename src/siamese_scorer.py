import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseScoringModule(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=1024):
        super().__init__()
        self.canonical_embedding = nn.Embedding(vocab_size, emb_dim)
        
        self.network = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

    def forward(self, dec_r, canonical_phonemes):
        can_emb = self.canonical_embedding(canonical_phonemes)
        
        batch_size, seq_len, dim = dec_r.shape
        
        dec_r_flat = dec_r.reshape(-1, dim)
        can_emb_flat = can_emb.reshape(-1, dim)
        
        out_pronounced = self.network(dec_r_flat)
        out_canonical = self.network(can_emb_flat)
        
        sim = F.cosine_similarity(out_pronounced, out_canonical, dim=-1)
        
        return sim.view(batch_size, seq_len)


class E2ERScorer(nn.Module):
    def __init__(self, apr_model, vocab_size, emb_dim=128):
        super().__init__()
        self.apr = apr_model
        self.siamese = SiameseScoringModule(vocab_size, emb_dim=emb_dim)

    def forward(self, input_values, labels, target_scores=None):
        ssl_features = self.apr.wav2vec2(input_values).last_hidden_state
        
        start_tokens = torch.full((labels.size(0), 1), self.apr.start_token_id, dtype=torch.long, device=labels.device)
        decoder_inputs = torch.cat([start_tokens, labels], dim=1)
        
        _, dec_r, _ = self.apr.pprm(ssl_features, decoder_inputs)
        
        predicted_scores = self.siamese(dec_r, labels)
        
        if target_scores is not None:
            loss = F.mse_loss(predicted_scores, target_scores)
            return loss, predicted_scores
            
        return predicted_scores