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
    wk: []const f32, // (layer, dim, n_kv_heads * head_dim)
    wv: []const f32, // (layer, dim, n_kv_heads * head_dim)
    wo: []const f32, // (layer, n_heads * head_dim, dim)
    wq_norm: []const f32, // (layer, head_dim)
    wk_norm: []const f32, // (layer, head_dim)
    // weights for ffn. w1 = up, w3 = gate, w2 = down
    w1: []const f32, // (layer, dim, hidden_dim)
    w2: []const f32, // (layer, hidden_dim, dim)
    w3: []const f32, // (layer, dim, hidden_dim)
    // final rmsnorm
    rms_final_weight: []const f32, // (dim,)
    // Same as token_embedding_table. GGUF has the final layer anyway
    wcls: []const f32,
};

const RunState = struct {
    // current wave of activations
    x: []const f32, // activation at current time stamp (dim,)
    xb: []const f32, // buffer (dim,)
    xb2: []const f32, // an additional buffer just for convenience (dim,)
    xb3: []const f32, // an additional buffer just for convenience (att_head_dim,)
    hb: []const f32, // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: []const f32, // buffer for hidden dimension in the ffn (hidden_dim,)
    q: []const f32, // query (att_head_dim,)
    k: []const f32, // key (dim,)
    v: []const f32, // value (dim,)
    att: []const f32, // buffer for scores/attention values (n_heads, seq_len)
};

pub fn main() void {}
