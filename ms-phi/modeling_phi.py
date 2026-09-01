from typing import List, Tuple
from dataclasses import dataclass

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PhiLMConfig:
    """Model the configuration from here: https://github.com/huggingface/transformers/blob/main/src/transformers/models/phi/configuration_phi.py"""

    vocab_size: int = 32064
    hidden_size: int = 3072
    intermediate_size: int = 8192
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: int = 10000.0
    bos_token_id: int = 1
    eos_token_id: int = 32000
    pad_token_id: int = 32000


class PhiCausalLM(nn.Module):
    def __init__(self, config: PhiLMConfig):
        super().__init__()

        self.max_position_embeddings = config.max_position_embeddings
        rope = Phi3RotaryPositionEmbedding(config)

        self.model = nn.ModuleDict(
            dict(
                embed_tokens=nn.Embedding(
                    config.vocab_size,
                    config.hidden_size,
                    padding_idx=config.pad_token_id,
                    dtype=torch.bfloat16,
                ),
                layers=nn.ModuleList(
                    [DecodeLayer(config, rope) for _ in range(config.num_hidden_layers)]
                ),
                norm=Phi3RMSNorm(config),
            )
        )

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=torch.bfloat16)

    @classmethod
    def from_pretrained(cls, model_id: str = 'microsoft/Phi-3.5-mini-instruct'):
        from transformers import AutoModelForCausalLM
        weights = AutoModelForCausalLM.from_pretrained(model_id)

        config = PhiLMConfig()
        model = cls(config)
        model.load_state_dict(weights.state_dict(), strict=False)
        return model

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward the input through the transformer layer, return logits.
        input shape (Batch_size, Sequence size)
        return shape (Batch_size, Sequence size, vocab_size)
        """
        _, S = x.size()
        assert S <= self.max_position_embeddings

        x = self.model.embed_tokens(x)

        for layer in self.model.layers:
            x = layer(x)

        x = self.model.norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 20,
        tokenizer=None,
        eos_token_id: int = 32000,
        return_tokens: bool = False,
    ):
        """Generate text given a prompt string and maximum number of tokens.

        Args:
            prompt (str): Input text prompt.
            max_tokens (int): Maximum number of answer tokens to generate.
            tokenizer: AutoTokenizer instance. If None, loaded from pretrained model_id.
            eos_token_id (int): EOS token ID to terminate generation. Defaults to 32000.
            return_tokens (bool): If True, returns torch.Tensor of token IDs instead of string.

        Returns:
            str: Detokenized generated answer text (or torch.Tensor if return_tokens=True).
        """
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3.5-mini-instruct')

        device = next(self.parameters()).device
        if isinstance(prompt, str):
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        else:
            input_ids = prompt.to(device)

        if eos_token_id is None and hasattr(tokenizer, "eos_token_id"):
            eos_token_id = tokenizer.eos_token_id

        generated_tokens = []
        curr_input_ids = input_ids

        for _ in range(max_tokens):
            logits = self.forward(curr_input_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            token_id = next_token.item()
            generated_tokens.append(token_id)
            if eos_token_id is not None and token_id == eos_token_id:
                break
            curr_input_ids = torch.cat([curr_input_ids, next_token], dim=-1)

        if return_tokens:
            return torch.tensor(generated_tokens, dtype=torch.long, device=device)

        return tokenizer.decode(generated_tokens)


class Phi3RMSNorm(nn.Module):
    """Phi3 uses a Llama RMS Norm: layer normalizes by dividing the stddev of the sequence."""

    def __init__(self, config: PhiLMConfig):
        super().__init__()

        self.rms_norm_eps = config.rms_norm_eps
        self.weight = nn.Parameter(torch.ones(config.hidden_size, dtype=torch.bfloat16))

    def forward(self, x):
        # TODO: consider upscaling for better float precision

        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.rms_norm_eps).to(torch.bfloat16)

        return self.weight * x


class Phi3RotaryPositionEmbedding(nn.Module):
    """ See more about RoPE at:
    - https://github.com/huggingface/transformers/blob/main/src/transformers/models/phi3/modeling_phi3.py
    - https://nn.labml.ai/transformers/rope/index.html
    - https://fleetwood.dev/posts/you-could-have-designed-SOTA-positional-encoding
    """
    def __init__(self, config: PhiLMConfig, dtype: torch.dtype = torch.bfloat16):
        super().__init__()

        self.dim = config.hidden_size // config.num_attention_heads
        self.seq_size = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.cache = None
        cos, sin = self._init_cache(dtype=dtype)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _init_cache(self, dtype: torch.dtype = torch.bfloat16) -> Tuple[torch.Tensor, torch.Tensor]:
        """ return cos and sin cache as shape (seq, dim)
        """
        # force float32 to avoid precision issues
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, self.dim, 2, dtype=torch.int64).to(torch.float32) / self.dim)
        )
        position_ids = torch.arange(0, self.seq_size).float()

        cache = torch.outer(position_ids, inv_freq)
        cache = torch.cat((cache, cache), dim=-1)
        self.cache = cache

        return cache.cos().to(dtype), cache.sin().to(dtype)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x[..., :x.shape[-1] // 2]
        x1 = x[..., x.shape[-1] // 2:]

        return torch.cat((-x1, x0), dim=-1)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ assume the x input is of shape (bs, head, seq, head_dim)
        """
        _, _, s, head_dim = x.shape

        cos = self.cos[:s, :].view(1, 1, s, head_dim).to(device=x.device, dtype=x.dtype)
        sin = self.sin[:s, :].view(1, 1, s, head_dim).to(device=x.device, dtype=x.dtype)

        return x * cos + self.rotate_half(x) * sin


class DecodeLayer(nn.Module):

    def __init__(self, config: PhiLMConfig, rope: "Phi3RotaryPositionEmbedding"):
        super().__init__()

        self.input_layernorm = Phi3RMSNorm(config)
        self.post_attention_layernorm = Phi3RMSNorm(config)

        self.self_attn = AttentionLayer(config, rope)
        self.mlp = MLP(config)

    def forward(self, x) -> torch.Tensor:
        # go through self attention layer
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = residual + x

        # go through mlp layer
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class AttentionLayer(nn.Module):

    def __init__(self, config: PhiLMConfig, rope: "Phi3RotaryPositionEmbedding"):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.heads = config.num_attention_heads
        self.qkv_proj = nn.Linear(
            config.hidden_size, 3 * config.hidden_size, bias=False, dtype=torch.bfloat16,
        )
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=torch.bfloat16)

        self.rope = rope
        seq = config.max_position_embeddings
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(seq, seq, dtype=torch.bfloat16)).view(
                1, 1, seq, seq,
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input of shape (bs, seq, hidden)
        qkv = self.qkv_proj(x)

        # split qkv into shape of (bs, seq, 3 * hidden) into q, k, v of shape (bs, seq, hidden)
        q, k, v = torch.split(qkv, self.hidden_size, dim=-1)

        # transpose all into:
        #   shape (bs, seq, heads, head_size)
        #   shape (bs, heads, seq, head_size)
        # so that all subsequent operations are on each head
        head_size = self.hidden_size // self.heads
        input_shape = x.shape
        hidden_shape = (*x.shape[:-1], -1, head_size)

        q = q.contiguous().view(hidden_shape).transpose(1, 2)
        k = k.contiguous().view(hidden_shape).transpose(1, 2)
        v = v.contiguous().view(hidden_shape).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        # output shape (bs, head, seq, head_size)
        output = q @ k.transpose(-2, -1) / math.sqrt(head_size) # bs, head, seq, seq

        # apply attention mask
        seq = x.shape[1]
        output = output.masked_fill(self.mask[:, :, :seq, :seq] == 0, float('-inf'))
        output = torch.softmax(output, dim=-1)
        output = output @ v # bs, head, seq, head_size

        # output of shape (bs, seq, hidden)
        output = output.transpose(1, 2)  # bs, seq, head, head_size
        output = output.contiguous().view(input_shape)

        output = self.o_proj(output)
        return output


class MLP(nn.Module):
    """Phi MLP uses a gated activation that:
    - has a gate signal that controls the flow of information
    - uses silu for the MLP activation
    """

    def __init__(self, config: PhiLMConfig):
        super().__init__()

        self.gate_up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size * 2, bias=False, dtype=torch.bfloat16,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False, dtype=torch.bfloat16,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gate_up_proj(x)
        gate, up_states = x.chunk(2, dim=-1)

        act = nn.SiLU()

        up_states = up_states * act(gate)

        return self.down_proj(up_states)
