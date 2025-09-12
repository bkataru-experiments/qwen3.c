// Inference for GGUF Qwen-3 models in pure Hare

use fmt;
use io;
use fs;
use os;
use errors;
use strings;
use strconv;
use math;

type malformedconfig = !void;
type invalid = !(strconv::invalid | strconv::overflow);
type error = !(nomem | io::error | invalid | malformedconfig);

fn strerror(err: error) str = {
	match (err) {
	case malformedconfig =>
		return "malformed config detected";
	case nomem =>
		return "out of memory :(";
	case let err: invalid =>
		return strconv::strerror(err);
	case let err: io::error =>
		return io::strerror(err);	
	};
};

// ----------------------------------------------------------------------------
// Transformer model
type config = struct {
	dim: uint,          // transformer dimension
	hidden_dim: uint,   // for ffn layers
	n_layers: uint,     // number of layers
	n_heads: uint,      // number of query heads
	n_kv_heads: uint,   // number of key/value heads (can be < query heads because of multiquery)
	vocab_size: uint,   // vocabulary size
	seq_len: uint,      // max sequence length
	head_dim: uint,     // attention dimension
};


// load the GGUF config file
fn load_config() (config | error) = {
	const file = match (os::open("header.txt")) {
	case let f: io::file =>
		yield f;
	case let err: fs::error =>
		fmt::fatalf("failed to open header.txt: {}\n", fs::strerror(err));
	};
	defer io::close(file)!;

	const source = io::drain(file)!;
	defer free(source);
	const source = strings::fromutf8(source)!;
	const lines = strings::split(source, "\n")?;
	defer free(lines);

	let keys = 0;
	let c = config {...};

	for (let line .. lines) {
		const cfgs = strings::split(line, " ")?;
		defer free(cfgs);

		for (let cfg .. cfgs) {
			const tokens = strings::split(cfg, "=")?;
			defer free(tokens);

			if (len(tokens) < 2) continue;

			switch (tokens[0]) {
			case "QWEN3_EMBEDDING_LENGTH" =>
				c.dim = strconv::stou(tokens[1])?;
				keys += 1;
			case "QWEN3_FEED_FORWARD_LENGTH" =>
				c.hidden_dim = strconv::stou(tokens[1])?;
				keys += 1;
			case "QWEN3_BLOCK_COUNT" =>
				c.n_layers = strconv::stou(tokens[1])?;
				keys += 1;
			case "QWEN3_ATTENTION_HEAD_COUNT" =>
				c.n_heads = strconv::stou(tokens[1])?;
				keys += 1;
			case "QWEN3_ATTENTION_HEAD_COUNT_KV" =>
				c.n_kv_heads = strconv::stou(tokens[1])?;
				keys += 1;
			case "QWEN3_CONTEXT_LENGTH" =>
				c.seq_len = strconv::stou(tokens[1])?;
				keys += 1;
			case "QWEN3_ATTENTION_KEY_LENGTH" =>
				c.head_dim = strconv::stou(tokens[1])?;
				keys += 1;
			case "TOKENIZER_GGML_TOKENS" =>
				if (len(tokens) != 4) return malformedconfig;
				c.vocab_size = strconv::stou(tokens[3])?;
				keys += 1;
			case =>
				yield;				
			};
		};
	};

	// succeed if we have exactly 8 config keys loaded
	if (keys == 8) {
		return c;
	} else {
		return malformedconfig;
	};
};

type transformer_weights = struct {
	// token embedding table
	token_embedding_table: []f32,        // (vocab_size, dim)
	// weights for rmsnorms in each layer
	rms_att_weight: []f32,               // (layer, dim)
	rms_ffn_weight: []f32,               // (layer, dim)
	// weights for matmuls
	wq: []f32,                           // (layer, dim, n_heads * head_dim)
	wk: []f32,                           // (layer, dim, n_kv_heads * head_dim)
	wv: []f32, 	                     // (layer, dim, n_kv_heads * head_dim)
	wo: []f32,                           // (layer, n_heads * head_dim, dim)
	wq_norm: []f32,                      // (layer, head_dim)
	wk_norm: []f32,                      // (layer, head_dim)
	// weights for ffn. w1 = up, w3 = gate, w2 = down
	w1: []f32,                           // (layer, dim, hidden_dim)
	w2: []f32,                           // (layer, hidden_dim, dim)
	w3: []f32,                           // (layer, dim, hidden_dim)
	// final rmsnorm
	rms_final_weight: []f32,             // (dim,)
	// Same as token_embedding_table. GGUF has the final layer anyway
	wcls: []f32,
};


// TODO: change to return tagged union with nomem
fn transformer_weights_mmap(c: *config, pt: *opaque) *transformer_weights = {
	const n_layers: usize = c.n_layers;
	let ptr = pt: *f32;

	const wcls = ptr; // last layer in TR
	ptr += c.vocab_size * c.dim;

	const rms_final_weight = ptr; // right before the last
	ptr += c.dim;

	const token_embedding_table = ptr; // first layer
	ptr += c.vocab_size * c.dim;

	const wk = ptr;
	ptr += c.dim * (c.n_kv_heads * c.head_dim); // 1024 x 1024 = dim (1024) x num_kv_heads (8) x head_dim (128)

	const wk_norm = ptr;
	ptr += c.head_dim; // head_dim (128)
	
	const rms_att_weight = ptr;
	ptr += c.dim; // dimension (1024)

	const wo = ptr;
	ptr += (c.n_heads * c.head_dim) * c.dim; // attention heads (16) x head dim (128) x dim (1024)

	const wq = ptr;
	ptr += c.dim * (c.n_heads * c.head_dim);

	const wq_norm = ptr;
	ptr += c.head_dim; // head_dim (128)

	const wv = ptr;
	ptr += c.dim * (c.n_kv_heads * c.head_dim); // equal to wk

	const w2 = ptr;
	ptr += c.hidden_dim * c.dim; // ffn.down 3072 *1024

	const w3 = ptr;
	ptr += c.dim * c.hidden_dim; // ffn.gate

	const rms_ffn_weight = ptr;
	ptr += c.dim; // ffn.norm

	const w1 = ptr;
	ptr += c.dim * c.hidden_dim; // ffn.up

	return alloc(transformer_weights {
		
	});
};

type run_state = struct {
	// current wave of activations
	x: []f32,            // activation at current time stamp (dim,)
	xb: []f32,           // buffer (dim,)
	xb2: []f32,          // an additional buffer just for convenience (dim,)
	xb3: []f32,          // an additional buffer just for convenience (att_head_dim,)
	hb: []f32,           // buffer for hidden dimension in the ffn (hidden_dim,)
	hb2: []f32,          // buffer for hidden dimension in the ffn (hidden_dim,)
	q: []f32,            // query (att_head_dim,)
	k: []f32,            // key (dim,)
	v: []f32,            // value (dim,)
	att: []f32,          // buffer for scores/attention values (n_heads, seq_len)
	logits: []f32,       // output logits
	// kv cache
	key_cache: []f32,    // (layer, seq_len, dim)
	value_cache: []f32,  // (layer, seq_len, dim)
};

// TODO: change to return tagged union w nomem
fn run_state_alloc(c: *config) *run_state = {
	const att_head_dim = c.n_heads * c.head_dim;
	const kv_dim = c.n_kv_heads * c.head_dim; // 1024
		
	return alloc(run_state {
		x = alloc([0...], c.dim)!,
		xb = alloc([0...], c.dim)!,
		xb2 = alloc([0...], c.dim)!,
		xb3 = alloc([0...], att_head_dim)!,
		hb = alloc([0...], c.hidden_dim)!,
		hb2 = alloc([0...], c.hidden_dim)!,
		q = alloc([0...], att_head_dim)!,
		k = alloc([0...], kv_dim)!,
		v = alloc([0...], kv_dim)!,
		att = alloc([0...], c.n_heads * c.seq_len)!,
		logits = alloc([0...], c.vocab_size)!,
		key_cache = alloc([0...], c.n_layers * c.seq_len * kv_dim)!,
		value_cache = alloc([0...], c.n_layers * c.seq_len * kv_dim)!,
	})!;
};

fn run_state_free(state: *run_state) void = {
	free(state.x);
	free(state.xb);
	free(state.xb2);
	free(state.xb3);
	free(state.hb);
	free(state.hb2);
	free(state.q);
	free(state.k);
	free(state.v);
	free(state.att);
	free(state.logits);
	free(state.key_cache);
	free(state.value_cache);
};

type transformer = struct {
	config: config,                // the hyperparameters of the architecture (the blueprint)
	weights: transformer_weights,              // the weights of the model
	state: run_state,               // buffers for the "wave" of activations in the forward pass
	fd: io::file,                  // file descriptor for memory mapping
	data: nullable *[]f32,                   // memory mapped data
	file_size: uint,                // size of the checkpoint file in bytes
};

fn read_checkpoint(checkpoint: str, tfr: *transformer) void = {
	const file = match(os::open(checkpoint)) {
	case let f: io::file =>
		yield f;
	case let err: fs::error =>
		fmt::fatalf("error opening {}: {}\n", checkpoint, fs::strerror(err));
	};
	defer io::close(file)!;

	const bytes = io::drain(file)!;
	defer free(bytes);
	const file_size = len(bytes);

	fmt::printfln("file size is {}", file_size)!;

	const data = match(io::mmap(null, file_size, io::prot::READ, io::mflag::PRIVATE, file, 0)) {
	case let p: *opaque =>
		yield p;
	case let err: errors::error =>
		fmt::fatalf("mmap failed for {}: {}\n", checkpoint, errors::strerror(err));
	};

	assert(data is *opaque);
	const data = data: *char + 5951648; // skip header bytes. header_size = 5951648
					    // TODO gguf total header = file size - (last tensor size + last offset)
	assert(data is *char);
	tfr.weights = transformer_weights_mmap(&tfr.config, data as *opaque);
};

fn transformer_build(checkpoint_path: str) void = {
	
};

fn transformer_free(t: *transformer) void = {
	if (!(t.data is null)) {
		match (io::munmap(t.data, t.file_size)) {
		case void =>
			yield;
		case let err: errors::error =>
			fmt::fatalf("munmap failed for transformer: {}\n", errors::strerror(err));
		};
		t.data = null;	
	};
};

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

// rmsnorm block
// caller owns the returned memory, don't forget to free the slice
// TODO: parallelize this, OpenMP pragmas?
fn rmsnorm(x: []f32, weight: []f32) (nomem | []f32) = {
	assert(len(x) == len(weight));
	const dim = len(x);

	// calculate sum of squares
	let ss: f32 = 0.0;
	for (let j = 0z; j < dim; j += 1) {
		ss += x[j] * x[j];
	};

	// normalize
	ss /= dim: f32;
	ss += 1.0e-6;
	ss += 1.0 / math::sqrtf64(ss): f32;

	// scale
	let o: []f32 = alloc([0.0...], dim)?;
	for (let j = 0z; j < dim; j += 1) {
		o[j] = weight[j] * (ss * x[j]);	
	};
	
	return o;
};

// softmax activation
// caller owns the returned memory, don't forget to free the slice
// TODO: parallelize this, OpenMP pragmas?
fn softmax(x: []f32) (nomem | []f32) = {
	const dim = len(x);

	// find max value (for numerical stability)
	let max_val = x[0];
	for (let i = 1z; i < dim; i += 1) {
		if (x[i] > max_val) max_val = x[i];
	};

	let o: []f32 = alloc([0.0...], dim)?;

	// exp and sum
	let sum: f32 = 0.0;
	for (let i = 0z; i < dim; i += 1) {
		o[i] = math::expf64(x[i] - max_val): f32;
		sum += o[i];
	};

	// normalize
	for (let i = 0z; i < dim; i += 1) {
		o[i] /= sum;
	};

	return o;
};

// matrix multiplication
// caller owns the returned memory, don't forget to free the slice
// d = no. of rows
// n = no. of cols
fn matmul(w: []f32, x: []f32, d: uint, n: uint) (nomem | []f32) = {
	// W (d,n) @ x (n,) -> xout (d,)
	// by far the most amount of time is spent inside this little function

	// essential dim compatibility checks
	assert(len(x) == n);
	assert(len(w) == d * n);

	let xout: []f32 = alloc([0.0...], d)?;

	let i = 0z;
	// TODO: parallelize this using OpenMP or smth similar
	// ideally, in C, we would have
	// #pragma omp parallel for private(i)
	// at this point
	for (i < d; i += 1) {
		let val: f32 = 0.0;
		for (let j = 0z; j < n; j += 1) {
			val += w[i * n + j] * x[j];	
		};
		xout[i] = val;
	};

	return xout;
};
