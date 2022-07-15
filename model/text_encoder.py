import torch
from torch import nn
from _common import Transformer, LayerNorm


class TextEncoder(nn.Module):
    def __init__(self, transformer_width, transformer_layers, transformer_heads, context_length, vocab_size, embed_dim,
                 tea_transformer_width=None, is_student=True):
        super().__init__()
        self.context_length = context_length
        self.is_student = is_student
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        if is_student:
            self.embedding_projection = nn.Linear(transformer_width, tea_transformer_width)
            self.hidden_projection = nn.Linear(transformer_width, tea_transformer_width)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_text(self, text, only_last_state=True):
        embedding = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = embedding + self.positional_embedding
        # x = x.permute(1, 0, 2)  # NLD -> LND
        x, attention_maps, representations = self.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        if only_last_state:
            return x
        if self.is_student:
            for i in range(len(representations)):
                representations[i] = self.hidden_projection(representations[i])
            embedding = self.embedding_projection(embedding)

        return x, attention_maps, representations, embedding

    def forward(self, text, only_last_state=True):
        return self.encode_text(text, only_last_state)


if __name__ == '__main__':
    m = TextEncoder(256, 2, 8, 77, 300, 150, 768, True)
    inputs = torch.randint(low=0, high=300, size=(64, 77))
    outputs = m(inputs, False)
    print(len(outputs), outputs[0].shape)
