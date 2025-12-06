/// Inference for GGUF Qwen-3 models in pure Rust

// ----------------------------------------------------------------------------
// Transformer model
#[derive(Debug)]
struct Config {
    dim: usize,        // transformer dimension
    hidden_dim: usize, // for ffn layers
    n_layers: usize,   // number of layers
    n_heads: usize,    // number of query heads
    n_kv_heads: usize, // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: usize, // vocabulary size
    seq_len: usize,    // max sequence length
    head_dim: usize,   // attention dimension
}

#[derive(Debug)]
struct TransformerWeights {
    // token embedding table
    token_embedding_table: Box<[f32]>, // (vocab_size, dim)
    // weights for rmsnorms in each layer
    rms_att_weight: Box<[f32]>, // (layer, dim)
    rms_ffn_weight: Box<[f32]>, // (layer, dim)
    // weights for matmuls
    wq: Box<[f32]>,      // (layer, dim, n_heads * head_dim)
    wk: Box<[f32]>,      // (layer, dim, n_kv_heads * head_dim)
    wv: Box<[f32]>,      // (layer, dim, m_kv_heads * head_dim)
    wo: Box<[f32]>,      // (layer, n_heads * head_dim, dim)
    wq_norm: Box<[f32]>, // (layer, head_dim)
    wk_norm: Box<[f32]>, // (layer, head_dim)
    // weights for ffn. w1 = up, w3 = gate, w2 = down
    w1: Box<[f32]>, // (layer, dim, hidden_dim)
    w2: Box<[f32]>, // (layer, hidden_dim, dim)
    w3: Box<[f32]>, // (layer, dim, hidden_dim)
    // final rmsnorm
    rms_final_weight: Box<[f32]>, // (dim,)
    // Same as token_embedding_table. GGUF has the final layer anyway
    wcls: Box<[f32]>,
}

#[derive(Debug)]
struct RunState {
    // current wave of activations
    x: Box<[f32]>,   // activation at current time stamp (dim,)
    xb: Box<[f32]>,  // buffer (dim,)
    xb2: Box<[f32]>, // an additional buffer just for convenience
}

fn main() {
    println!("hello world");
}
