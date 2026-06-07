import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model
from pprm import PPRM


class FullAPRModel(nn.Module):
    def __init__(self, model_name, vocab_size, pad_token_id, start_token_id, end_token_id):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name, ignore_mismatched_sizes=True)
        
        self.wav2vec2.freeze_feature_encoder()

        self.wav2vec2.gradient_checkpointing_enable()
        
        self.pprm = PPRM(vocab_size=vocab_size, ssl_dim=self.wav2vec2.config.hidden_size)
        
        self.pad_token_id = pad_token_id
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        
        self.ctc_loss_fct = nn.CTCLoss(blank=pad_token_id, zero_infinity=True)
        self.att_loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.wav2vec2, "gradient_checkpointing_enable"):
            self.wav2vec2.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.wav2vec2, "gradient_checkpointing_disable"):
            self.wav2vec2.gradient_checkpointing_disable()

    def forward(self, input_values, labels=None, attention_mask=None):
        ssl_features = self.wav2vec2(input_values, attention_mask=attention_mask).last_hidden_state
        
        if labels is not None:
            labels_mask = labels != -100

            targets_ctc = labels.masked_fill(~labels_mask, self.pad_token_id)
            target_lengths = labels_mask.sum(-1)
            
            clean_labels = labels.masked_fill(~labels_mask, self.pad_token_id)
            start_tokens = torch.full((labels.size(0), 1), self.start_token_id, dtype=torch.long, device=labels.device)
            decoder_inputs = torch.cat([start_tokens, clean_labels], dim=1)
            
            end_tokens = torch.full((labels.size(0), 1), self.end_token_id, dtype=torch.long, device=labels.device)
            decoder_targets = torch.cat([clean_labels, end_tokens], dim=1)
            
            ctc_logits, _, att_logits = self.pprm(ssl_features, decoder_inputs)
            
            log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
            if attention_mask is not None:
                input_lengths = self.wav2vec2._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
            else:
                input_lengths = torch.full((labels.size(0),), log_probs.size(0), dtype=torch.long, device=labels.device)
                
            ctc_loss = self.ctc_loss_fct(log_probs, targets_ctc, input_lengths, target_lengths)
            
            target_mask_with_end = torch.cat([labels_mask, torch.ones((labels.size(0), 1), dtype=torch.bool, device=labels.device)], dim=1)
            decoder_targets = decoder_targets.masked_fill(~target_mask_with_end, -100)

            att_loss = self.att_loss_fct(att_logits.reshape(-1, att_logits.size(-1)), decoder_targets.view(-1))
            
            return (0.2 * ctc_loss + 0.8 * att_loss, ctc_logits)
            
        dummy_inputs = torch.full((input_values.size(0), 1), self.start_token_id, dtype=torch.long, device=input_values.device)
        ctc_logits, _, _ = self.pprm(ssl_features, dummy_inputs)
        return (ctc_logits,)