from __future__ import annotations

from collections import defaultdict
import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    raise NotImplementedError


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    raise NotImplementedError


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    raise NotImplementedError


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    raise NotImplementedError


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    raise NotImplementedError


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    raise NotImplementedError


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    raise NotImplementedError


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    raise NotImplementedError


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    raise NotImplementedError


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError

class Tokenizer:
    """
    Deliverable: Implement a Tokenizer class that, given a vocabulary and a list of merges, encodes
    text into integer IDs and decodes integer IDs into text. Your tokenizer should also support user-provided
    special tokens (appending them to the vocabulary if they aren’t already there). We recommend the
    following interface:



    def decode(self, ids: list[int]) -> str Decode a sequence of token IDs into text.
    To test your Tokenizer against our provided tests, you will first need to implement the test adapter
    at [adapters.get_tokenizer]. Then, run uv run pytest tests/test_tokenizer.py. Your implementation should be able to pass all tests.
    """

    def __init__(self, vocab, merges, special_tokens=None):
        #     Construct a tokenizer from a given
        # vocabulary, list of merges, and (optionally) a list of special tokens. This function should accept
        # the following parameters:
        # vocab: dict[int, bytes]
        # merges: list[tuple[bytes, bytes]]
        # special_tokens: list[str] | None = None
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.token_to_id = {v: k for k, v in vocab.items()}

    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        # Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        # (in the same format that your BPE training code output) and (optionally) a list of special
        # tokens. This method should accept the following additional parameters:
        # vocab_filepath: str
        # merges_filepath: str
        # special_tokens: list[str] | None = None
        vocab = {}
        merges = []
        with open(vocab_filepath, 'r') as vf:
            for line in vf:
                token_id_str, token_bytes_str = line.strip().split('\t')
                token_id = int(token_id_str)
                token_bytes = bytes.fromhex(token_bytes_str)
                vocab[token_id] = token_bytes
        with open(merges_filepath, 'r') as mf:
            for line in mf:
                token1_str, token2_str = line.strip().split('\t')
                token1 = bytes.fromhex(token1_str)
                token2 = bytes.fromhex(token2_str)
                merges.append((token1, token2))
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        # Encode an input text into a sequence of token IDs.
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        def split_with_special_tokens(text: str, special_tokens: list[str]) -> list[tuple[str, bool]]:
            """Split text on special tokens, returning (chunk, is_special) tuples.

            Preserves special tokens in the output with is_special=True.
            """
            if not special_tokens:
                return [(text, False)] if text else []

            # Sort special tokens by length (longest first) to handle overlapping tokens
            sorted_tokens = sorted(special_tokens, key=len, reverse=True)

            # Build pattern with capturing group to keep the special tokens
            pattern = "(" + "|".join(re.escape(tok) for tok in sorted_tokens) + ")"

            result = []
            parts = re.split(pattern, text)
            special_set = set(special_tokens)

            for part in parts:
                if part:  # Skip empty strings
                    if part in special_set:
                        result.append((part, True))
                    else:
                        result.append((part, False))

            return result

        def encode_chunk(chunk_text: str) -> list[int]:
            """Encode a single chunk (not a special token) into token IDs."""
            chunk_token_ids = []

            for match in re.finditer(PAT, chunk_text):
                # Start with individual bytes for each pre-token
                pre_token_bytes = match.group(0).encode("utf-8")

                # Each byte becomes its own token initially
                # Need to look up each byte in token_to_id to get its token ID
                token_ids_for_pretoken = []
                for byte_val in pre_token_bytes:
                    byte_as_bytes = bytes([byte_val])
                    token_id = self.token_to_id.get(byte_as_bytes)
                    if token_id is not None:
                        token_ids_for_pretoken.append(token_id)

                # Apply merges in order to combine bytes
                for merge in self.merges:
                    i = 0
                    while i < len(token_ids_for_pretoken) - 1:
                        # Get the bytes for current pair
                        first_bytes = self.vocab[token_ids_for_pretoken[i]]
                        second_bytes = self.vocab[token_ids_for_pretoken[i + 1]]

                        if (first_bytes, second_bytes) == merge:
                            merged_bytes = first_bytes + second_bytes
                            merged_token_id = self.token_to_id.get(merged_bytes)
                            if merged_token_id is not None:
                                token_ids_for_pretoken[i] = merged_token_id
                                del token_ids_for_pretoken[i + 1]
                                continue  # Stay at same index to check for more merges
                        i += 1

                chunk_token_ids.extend(token_ids_for_pretoken)

            return chunk_token_ids

        # Main encoding logic
        token_ids = []

        for chunk, is_special in split_with_special_tokens(text, self.special_tokens):
            if is_special:
                # Special token - look up directly
                special_token_bytes = chunk.encode("utf-8")
                special_token_id = self.token_to_id.get(special_token_bytes)
                if special_token_id is not None:
                    token_ids.append(special_token_id)
            else:
                # Regular text - encode with BPE
                token_ids.extend(encode_chunk(chunk))

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # Given an iterable of strings (e.g., a Python file handle), return a generator
        # that lazily yields token IDs. This is required for memory-efficient tokenization
        # of large files that we cannot directly load into memory.
        for line in iterable:
            token_ids = self.encode(line)
            for token_id in token_ids:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        # Decode a sequence of token IDs into text.
        tokens = [self.vocab[token_id] for token_id in ids]
        text = b''.join(tokens).decode('utf-8', errors='ignore')
        return text
        # raise NotImplementedError

def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab, merges, special_tokens=special_tokens)
    # raise NotImplementedError

from itertools import takewhile
from functools import reduce

def initialize_vocab(
    special_tokens: list[str] | None = None,
) -> dict[int, bytes]:
    """
    Initialize BPE vocabulary and prepare for merges and training.
    
    Args:
        special_tokens: A list of special token strings to be added to the vocabulary.
    Returns:
        A dictionary mapping integer token IDs to bytestring tokens.
            - merges: A list of tuples, each tuple containing two bytestrings
    """
    # Initialize the vocabulary with all possible byte values (0-255) as individual tokens.
    vocab = {i: bytes([i]) for i in range(256)}
    # Add special tokens
    for i, token in enumerate(special_tokens or []):
        vocab[256 + i] = token.encode("utf-8")

    return vocab

from functools import reduce
from itertools import takewhile

def apply_merge_functional(token_tuple: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """Replace all occurrences of pair in token_tuple using functional approach."""
    
    def process(acc_and_i, _):
        """Accumulator: (merged_list, index)"""
        merged, i = acc_and_i
        
        if i >= len(token_tuple):
            return (merged, i)
        
        if i < len(token_tuple) - 1 and token_tuple[i:i+2] == pair:
            return (merged + [pair[0] + pair[1]], i + 2)
        else:
            return (merged + [token_tuple[i]], i + 1)
    
    merged, _ = reduce(
        process,
        iter(range(len(token_tuple))),  # Dummy iterable
        ([], 0)  # Initial accumulator: (merged, index)
    )
    
    return tuple(merged)


def apply_merge(token_tuple: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """Replace all occurrences of pair in token_tuple."""
    if len(token_tuple) < 2:
        return token_tuple
    
    new_tokens = []
    i = 0
    while i < len(token_tuple):
        if i < len(token_tuple) - 1 and token_tuple[i] == pair[0] and token_tuple[i+1] == pair[1]:
            new_tokens.append(pair[0] + pair[1])
            i += 2
        else:
            new_tokens.append(token_tuple[i])
            i += 1
    
    return tuple(new_tokens)


from itertools import takewhile
from functools import reduce
import regex as re

def create_pre_tokenizer(input_path: str | os.PathLike, pattern: str, special_tokens: list[str]):
    """
    Create a closure that pre-tokenizes an input file.
    
    Args:
        input_path: Path to the input file
        pattern: Regex pattern for tokenization
    
    Returns:
        A function that returns pre_token_counts when called
    """
    def split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
        """Split text on special tokens, returning chunks between them."""
        if not special_tokens:
            return [text]
        
        # Build pattern: escape special chars, join with |
        pattern = "|".join(re.escape(tok) for tok in special_tokens)
        
        # Split and filter empty strings
        return [chunk for chunk in re.split(pattern, text) if chunk]

    def pre_tokenize() -> dict[tuple[bytes, ...], int]:
        pre_token_counts = defaultdict(int)
        with open(input_path, 'r') as f:
            text = f.read()  # Read entire file
            for chunk in split_on_special_tokens(text, special_tokens):
                for match in re.finditer(pattern, chunk):
                    token_bytes = tuple(bytes([b]) for b in match.group(0).encode("utf-8"))
                    pre_token_counts[token_bytes] += 1
        return pre_token_counts
    
    return pre_tokenize


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # BPE training using functional programming constructs.
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # Create pre-tokenizer closure and get token counts
    pre_tokenizer = create_pre_tokenizer(input_path, PAT, special_tokens=special_tokens)
    pre_token_counts = pre_tokenizer()

    vocab = initialize_vocab(special_tokens=special_tokens)

    num_merges = vocab_size - len(vocab)

    merges = []

    for _ in range(num_merges):
        # Count pairs
        pair_counts = defaultdict(int)
        for token_tuple, count in pre_token_counts.items():
            for i in range(len(token_tuple) - 1):
                pair = (token_tuple[i], token_tuple[i+1])
                pair_counts[pair] += count
        
        if not pair_counts:
            break
        
        # Find best pair
        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
        
        # Update vocab
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        
        # Record merge
        merges.append(best_pair)
        
        # Update pre_token_counts
        pre_token_counts = {
            # apply_merge_functional(token_tuple, best_pair): count
            apply_merge(token_tuple, best_pair): count
            for token_tuple, count in pre_token_counts.items()
        }

    return vocab, merges