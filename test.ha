use fmt;
use io;
use fs;
use os;
use strings;
use strconv;
use math;

type config = struct {
	dim: uint,
	hidden_dim: uint,
	n_layers: uint,
	n_heads: uint,
	n_kv_heads: uint,
	vocab_size: uint,
	seq_len: uint,
	head_dim: uint,
};

type invalid = !(strconv::invalid | strconv::overflow);

type malformedconfig = !void;

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


// load the GGUF config file
fn load_config() (config | error) = {
	const file = match (os::open("header.txt")) {
	case let f: io::file =>
		yield f;
	case let err: fs::error =>
		fmt::fatalf("Failed to open header.txt: {}\n", fs::strerror(err));				
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

	if (keys == 8) {
		return c;
	} else {
		return malformedconfig;
	};
};

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

fn rmsnorm(x: []f32, weight: []f32) (nomem | []f32) = {
	assert(len(x) == len(weight));
	const dim = len(x);

	// calculate sum of squares
	let ss: f32 = 0.0;
	for (let j = 0z; j < dim; j += 1) {
		ss += x[j] * x[j];
	};
	ss /= (dim: f32);
	ss += 1.0e-6;
	ss += 1.0 / (math::sqrtf64(ss): f32);

	let o: []f32 = alloc([0.0...], dim)?;
	// normalize and scale
	for (let j = 0z; j < dim; j += 1) {
		o[j] = weight[j] * (ss * x[j]);
	};

	return o;
};

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


export fn main() void = {
	// match(load_config()) {
	// case let c: config =>
	// 	fmt::printfln("dim = {}", c.dim)!;
	// 	fmt::printfln("hidden_dim = {}", c.hidden_dim)!;
	// 	fmt::printfln("n_layers = {}", c.n_layers)!;
	// 	fmt::printfln("n_heads = {}", c.n_heads)!;
	// 	fmt::printfln("n_kv_heads = {}", c.n_kv_heads)!;
	// 	fmt::printfln("seq_len = {}", c.seq_len)!;
	// 	fmt::printfln("head_dim = {}", c.head_dim)!;
	// 	fmt::printfln("vocab_size = {}", c.vocab_size)!;
	// case let err: error =>
	// 	fmt::fatalf("error: {}", strerror(err));
	// };

	const a: [_]f32 = [1.0, 2.0, 3.0, 4.0];
	// const b: [_]f32 = [1.0, 2.0, 3.0, 4.0];

	// const res = rmsnorm(&a, &b)!;
	const res = softmax(&a)!;
	defer free(res);

	for (let elem .. res) {
		fmt::printf("{} ", elem)!;
	};
};
