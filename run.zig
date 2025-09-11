/// Inference for GGUF Qwen-3 models in pure Zig
const std = @import("std");

// ----------------------------------------------------------------------------
// Transformer model
const Config = struct {
    dim: usize, // transformer dimension
    hidden_dim: usize, // for ffn layers
    n_layers: usize, // number of layers
    n_heads: usize, // number of query heads
    n_kv_heads: usize, // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: usize, // vocabulary size
    seq_len: usize, // max sequence length
    head_dim: usize, // attention dimension
};

const TransformerWeights = struct {
    // token embedding table
    token_embedding_table: []const f32, // (vocab_size, dim)
    // weights for rmsnorms in each layer
    rms_att_weight: []const f32, // (layer, dim)
    rms_ffn_weight: []const f32, // (layer, dim)
    // weights for matmuls
    wq: []const f32, // (layer, dim, n_heads * head_dim)
};

pub fn main() void {}
