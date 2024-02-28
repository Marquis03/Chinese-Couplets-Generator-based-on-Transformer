import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens) * math.sqrt(self.emb_size)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CoupletsTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super(CoupletsTransformer, self).__init__()
        self.name = "CoupletsTransformer"
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_embedding = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(d_model, vocab_size)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, padding_value=0):
        src_embed = self.token_embedding(src)  # [batch_size, src_len, embed_dim]
        src_embed = self.pos_embedding(src_embed)  # [batch_size, src_len, embed_dim]
        tgt_embed = self.token_embedding(tgt)  # [batch_size, tgt_len, embed_dim]
        tgt_embed = self.pos_embedding(tgt_embed)  # [batch_size, tgt_len, embed_dim]

        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(-1)).to(
            tgt.device
        )
        src_key_padding_mask = src == padding_value  # [batch_size, src_len]
        tgt_key_padding_mask = tgt == padding_value  # [batch_size, tgt_len]

        outs = self.transformer(
            src=src_embed,
            tgt=tgt_embed,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )  # [batch_size, tgt_len, embed_dim]
        logits = self.fc(outs)  # [batch_size, tgt_len, vocab_size]
        return logits

    def encoder(self, src):
        src_embed = self.token_embedding(src)
        src_embed = self.pos_embedding(src_embed)
        memory = self.transformer.encoder(src_embed)
        return memory

    def decoder(self, tgt, memory):
        tgt_embed = self.token_embedding(tgt)
        tgt_embed = self.pos_embedding(tgt_embed)
        outs = self.transformer.decoder(tgt_embed, memory=memory)
        return outs

    def generate(self, text, vocab):
        self.eval()
        device = next(self.parameters()).device
        max_len = len(text)
        src = torch.LongTensor(vocab(list(text))).unsqueeze(0).to(device)
        memory = self.encoder(src)
        l_out = [vocab.BOS]
        for i in range(max_len):
            tgt = torch.LongTensor(vocab(l_out)).unsqueeze(0).to(device)
            outs = self.decoder(tgt, memory)
            prob = self.fc(outs[:, -1, :])
            next_token = vocab.to_tokens(prob.argmax(1).item())
            if next_token == vocab.EOS:
                break
            l_out.append(next_token)
        return "".join(l_out[1:])


if __name__ == "__main__":
    vocab_size = 1000
    batch_size = 32
    model = CoupletsTransformer(vocab_size=vocab_size)
    src = torch.randint(0, 10, (batch_size, 10))
    tgt = torch.randint(0, 10, (batch_size, 11))
    logits = model(src, tgt)
    print(logits.shape)
