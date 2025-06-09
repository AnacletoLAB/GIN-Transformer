import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from pos_encoding import PositionalEncoding
from tokenizer import tokenizer

class NucConfig(PretrainedConfig):
    model_type = "nucformer"
    def __init__(self, vocab_size=6, d_model=128, nhead=4,
                 num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=512, dropout=0.1, max_len=200, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size, self.d_model = vocab_size, d_model
        self.nhead, self.dim_feedforward = nhead, dim_feedforward
        self.num_encoder_layers, self.num_decoder_layers = num_encoder_layers, num_decoder_layers
        self.dropout, self.max_len = dropout, max_len

class NucTransformer(PreTrainedModel):
    config_class = NucConfig
    def __init__(self, config: NucConfig):
        super().__init__(config)
        self.embed_src = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_tgt = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_enc = PositionalEncoding(config.d_model, config.max_len)
        self.transformer = nn.Transformer(
            d_model=config.d_model, nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout, batch_first=True
        )
        self.out = nn.Linear(config.d_model, config.vocab_size)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        src = self.pos_enc(self.embed_src(input_ids))
        tgt_in = labels[:,:-1]; tgt = self.pos_enc(self.embed_tgt(tgt_in))
        memory = self.transformer.encoder(src)
        dec = self.transformer.decoder(
            tgt, memory,
            tgt_mask=self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        )
        logits = self.out(dec)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            vocab_size = self.config.vocab_size
            loss = loss_fct(
                logits.view(-1, vocab_size),
                labels[:, 1:].reshape(-1)
            )
            loss = loss.unsqueeze(0)
        return {"loss": loss, "logits": logits}
