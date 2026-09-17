import torch
import torch.nn as nn
from attention import MultiHeadAttention

class TokenEmbedding(nn.Module):
    """
    Learnable lookup table: token_id -> vector of size n_embed
    Internally just a matrix of shape [vocab_size, n_embed].
    Each row is the embedding for one token.
    """
    def __init__(self, vocab_size, n_embed):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embed) # [VOCAB_SIZE, N_EMBED] matrix 
        # initilaized with values (0, 1) mean ~= 0, std.dev ~= 1

    def forward(self, idx):
        # idx shape: [B, T] (batch size, sequence length)
        # output : [B, T, n_embed] <- simple lookup for B * T tokens in idx.
        return self.embedding(idx)
    

class PositionalEmbedding(nn.Module):
    """
    Learnable lookup table: position_index -> vector of size n_embed
    Same structure as token embedding, but indexes by position, not token_id.
    """
    def __init__(self, block_size, n_embed):
        super().__init__()
        self.embedding = nn.Embedding(block_size, n_embed)

    def forward(self, T):
        # T: current sequence length / context window - for which model is trained & evaluated.
        # This parameter is a major deciding factor of model size (XS, S, M etc)
        # positions: [0, 1, 2, 3, ...., T-1]
        positions = torch.arange(T, device=self.embedding.weight.device)  # shape: [T]
        return self.embedding(positions)  # shape: [T, n_embed]
    

class GPTEmbedding(nn.Module):
    """
    Full embedding layer: combines token + positional embeddings.
    This is the first thing every input passes through.
    """
    def __init__(self, vocab_size, block_size, n_embed, dropout = 0.1):
        super().__init__()
        self.token_embed = TokenEmbedding(vocab_size, n_embed)
        self.pos_embed = PositionalEmbedding(block_size, n_embed)
        # *********** Dropout **************
        # prevents overfitting, during training it drops 0.xx fraction
        # of neurons. Each neuron has the probability (0 to 0.xx) of being set
        # to zero for current step. Does 2 things:
        # 1. Avoids co-adaptation: between neurons; w/o dropout neurons can develop 
        #    reliance on each other to fix their mistakes.
        # 2. Scaling: takes care of scaling by multiplying with * 1/0.xx. Because 
        #    dropout is turned off during inference -> all neurons are active. Signals
        #    flowing through the network would be XX% higher than training. Mismatch
        #    will break everything.
        # *************************************
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx):
        # idx shape: [B, T]
        B, T = idx.shape

        tok = self.token_embed(idx)   # [B, T, n_embed]
        pos = self.pos_embed(T)       # [T, n_embed]

        # pos broadcast across batch dimension automatically
        # [B, T, n_embed] + [T, n_embed] = [B, T, n_embed]
        x = tok + pos

        return self.dropout(x)
    

class FFN(nn.Module):
    """
    Feed Forward Network -> applied to each token independently (test - independently).
    2 linear layers with GeLU activation between them (wtf is that - LOL).
    Inner dimension is 4x n_embed -> standard transformer convention (wtf is that again - LOLLLLZZ)

    Takes:      [B, T, n_embed]
    Returns:    [B, T, n_embed] <- cool atleast stick to n_embed, but god knows what transformation now? 
    """

    def __init__(self, n_embed, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            # ***** nn.Linear(in, out, bias=True) ************
            # linear = nn.Linear(in, out);  learned weight matrix [out, in]
            # linear(x) applied as x @ weight.T (i.e. only transforms last dimension)
            # x.shape = [B, T, n_embed]; nn.Linear(n_embed, 4 * n_embed)
            # linear(x) = [B, T, n_emebd] @ [n_embed, 4 * embed] -> [B, T, 4 * embed]
            # ****** x @ linear.weight.T + linear.bias ********
            nn.Linear(n_embed, 4 * n_embed), # expand to 4x width -> why?
            nn.GELU(), # activation (smooth, non-zero for negatives -> let the negatives pass through)
            nn.Linear(4 * n_embed, n_embed), # project back down -> what's happening??
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x) # [B, T, n_embed] -> [B, T, n_embed]
    

class Block(nn.Module):
    """
    Full transformer block.
    Pre-LayerNorm architecture: LayerNorm applied before attention and FFN 
    # LayerNorm is applied to avoid drifts, exponential bloats in activations 
    # (values in embed vectors) as they go through several stages/layers of mult+add.
    Residual connections after both sub-layers.

    Takes:      x of shape [B, T, n_embed]
    Returns:    x of shape [B, T, n_embed]
    """

    def __init__(self, n_embed, n_heads, block_size, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = MultiHeadAttention(n_embed, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ffn = FFN(n_embed, dropout)

    def forward(self, x):
        # Pre-LN + residual connection around attention
        # Learns relationships/connection weights between the tokens.
        x = x + self.attn(self.ln1(x))

        # Pre-LN + residual connection around FFN
        # Learns the deeper knowledge per token independently.
        x = x + self.ffn(self.ln2(x))
        return x
    

class GPT(nn.Module):
    """
    Full GPT language model.

    Takes: ids of range [B, T] - token ids of batch size B and sequence length T
    Returns: logits [B, T, vocab_size] and optionally loss (perhaps scoring all token in vocab as next token)
    """

    def __init__(self, vocab_size, block_size, n_embed, n_heads, n_layers, dropout=0.0):
        super().__init__()
        self.block_size = block_size

        self.transformer = nn.ModuleDict({
            'embedding' : GPTEmbedding(vocab_size, block_size, n_embed, dropout),
            'blocks'    : nn.Sequential(*[
                Block(n_embed, n_heads, block_size, dropout)
                for _ in range(n_layers)
            ]),
            'ln_f'      : nn.LayerNorm(n_embed)
        })

        # Output head - projects vector [n_embed] -> score [vocab_size]
        # [n_embed] @ [n_embed, vocab_size] -> [vocab_size]
        # 1 64-dim vector -> 65 scores, one per token
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

        # Weights tying - share embedding and lm_head weights
        # * Token embedding weight matrix [vocab_size, n_embed] and
        # * lm_head matrix [vocab_size, n_embed] are the same object (physical matrix) in memory
        # * Save parameters, halves the memory (if not) and empirically improves the performance
        # * conceptually, both are essentially answering "what does token 'c' look like in n_embed
        # dimensional space.
        self.transformer['embedding'].token_embed.embedding.weight = \
            self.lm_head.weight
        
        # Intitalize weights (recursively  called for all sduless/sub-modules); 
        # if not initialized properly the training will crash.
        self.apply(self._init_weights) # ???

        print(f"GPT initialized - {self.count_params()/1e6:2f}M parameters")

    def _init_weights(self, module):
        """
        Initialize weights following GPT-2 paper.
        Linear and Embeddings layers: normal distribution std:0.02
        Bias: zero initialized
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward_repr_post_lnf(self, idx):
      x = self.forward_repr_pre_lnf(idx)     # [B, T, n_embed]
      return self.transformer['ln_f'](x)

    def forward_repr_pre_lnf(self, idx):
      x = self.transformer['embedding'](idx)
      return self.transformer['blocks'](x)

    def forward_repr_at_site(self, site, idx):
      if site == 'final_post_lnf':
        return self.forward_repr_post_lnf(idx)
      if site == 'final_pre_lnf':
        return self.forward_repr_pre_lnf(idx)
      if site.startswith('block'):
        i = int(site[5:])
        x = self.transformer['embedding'](idx)
        x = self.transformer['blocks'][:i](x)
        return x
      raise ValueError(site)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, \
            f"Sequence length {T} exceeds block_size {self.block_size}"
        
        # Embedding
        x = self.forward_repr_post_lnf(idx)  # [B, T, n_embed]

        # Project to vocabulary
        logits = self.lm_head(x)                # [B, T, vocab_size]

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            # CrossEntropyLoss expects [B*T, V] and [B, T]
            loss = nn.functional.cross_entropy(
                logits.view(B * T, V),
                targets.view(B * T)
            )
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate max_new_tokens tokens autoregressively.
        idx: starting context [B, T]
        """
        for _ in range(max_new_tokens):
            # Crop to block_size if sequence too long
            idx_cond = idx[:, -self.block_size:]  # LOL - help me understand this?

            # Forward pass - no targets so loss computed
            logits, _ = self(idx_cond)

            # Take logits at last position - next token prediction
            logits = logits[:, -1, :]  # [B, vocab_size]

            # Apply temperature - higher = more random, lower = more deterministic
            logits = logits / temperature

            # optional top-K sampling -- zero all out except Top K logits
            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, -1:]] = float('-inf')

            # Softmax to probabilities
            probs = nn.functional.softmax(logits, dim=-1)

            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1) # [B, 1]

            # Append to sequence
            idx = torch.cat([idx, next_token], dim=1)  # [B, T+1]
        
        return idx