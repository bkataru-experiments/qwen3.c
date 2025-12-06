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
    x: Box<[f32]>,      // activation at current time stamp (dim,)
    xb: Box<[f32]>,     // buffer (dim,)
    xb2: Box<[f32]>,    // an additional buffer just for convenience (dim,)
    xb3: Box<[f32]>,    // an additional buffer just for convenience (att_head_dim,)
    hb: Box<[f32]>,     // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: Box<[f32]>,    // buffer for hidden dimension in the ffn (hidden_dim,)
    q: Box<[f32]>,      // query (att_head_dim,)
    k: Box<[f32]>,      // key (dim,)
    v: Box<[f32]>,      // value (dim,)
    att: Box<[f32]>,    // buffer for scores/attention values (n_heads, seq_len)
    logits: Box<[f32]>, // output logits
    // kv cache
    key_cache: Box<[f32]>,   // (layer, seq_len, dim)
    value_cache: Box<[f32]>, // (layer, seq_len, dim)
}

#[derive(Debug)]
struct Transformer {
    config: Config,              // the hyperparameters of the architecture (the blueprint)
    weights: TransformerWeights, // the weights of the model
    state: RunState,             // buffers for the "wave" of activations in the forward pass
    fd: i32,                     // file descriptor for memory mapping
    data: Box<[f32]>,            // memory mapped data pointer
    file_size: isize,            // size of the checkpoint file in bytes
}

impl RunState {
    pub fn calloc(p: Config) -> Self {
        let att_head_dim = p.n_heads * p.head_dim;

        Self {
            x: vec![0.0; p.dim].into_boxed_slice(),
            xb: vec![0.0; p.dim].into_boxed_slice(),
            xb2: vec![0.0; p.dim].into_boxed_slice(),
            xb3: vec![0.0; att_head_dim].into_boxed_slice(),
        }
    }
}

fn main() {
    println!("hello world");
}
