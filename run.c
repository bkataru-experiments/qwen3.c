/* Inference for GGUF Qwen-3 models in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdbool.h>
#include <ctype.h>
#include <unistd.h>
#include <sys/mman.h>

// ----------------------------------------------------------------------------
// Transformer model
typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size
    int seq_len; // max sequence length
    int head_dim; // attention dimension
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table; // (vocab_size, dim)
    // weights for rmsnorms in each layer
    float* rms_att_weight; // (layer, dim)
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls
    float* wq; // (layer, dim, n_heads * head_dim)
    float* wk; // (layer, dim, n_kv_heads * head_dim)
    float* wv; // (layer, dim, n_kv_heads * head_dim)
    float* wo; // (layer, n_heads * head_dim, dim)
    float* wq_norm; // (layer, head_dim)
    float* wk_norm; // (layer, head_dim)
    // weights for ffn. w1 = up, w3 = gate, w2 = down
    float* w1; // (layer, dim, hidden_dim)
    float* w2; // (layer, hidden_dim, dim)
    float* w3; // (layer, dim, hidden_dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // Same as token_embedding_table. GGUF has the final layer anyway
    float* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float* x; // activation at current time stamp (dim,)
    float* xb; // buffer (dim,)
    float* xb2; // an additional buffer just for convenience (dim,)
    float* xb3; // an additional buffer just for convenience (att_head_dim,)
    float* hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float* hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float* q; // query (att_head_dim,)
    float* k; // key (dim,)
    float* v; // value (dim,)
    float* att; // buffer for scores/attention values (n_heads, seq_len)
    float* logits; // output logits
    // kv cache
    float* key_cache; // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int att_head_dim = p->n_heads * p->head_dim;
    int kv_dim = p->n_kv_heads * p->head_dim; // 1024

    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->xb3 = calloc(att_head_dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(att_head_dim, sizeof(float));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));

    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->xb3 || !s->hb || !s->hb2 || !s->q || !s->k || !s->v || !s->att || !s->logits || !s->key_cache || !s->value_cache) {
        fprintf(stderr, "calloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->xb3);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

// Map GGUF layers to transformer weights
void memory_map_weights(TransformerWeights* w, Config* p, void* pt) {
    unsigned long long n_layers = p->n_layers;
    float* ptr = (float*) pt;

    w->wcls = ptr; // last layer in TR
    ptr += p->vocab_size * p->dim;
    w->rms_final_weight = ptr; // right before the last
    ptr += p->dim;
    w->token_embedding_table = ptr; // first layer
    ptr += p->vocab_size * p->dim;
    w->wk = ptr;
    ptr += p->dim * (p->n_kv_heads * p->head_dim); // 1024 x 1024 = dim (1024) x num_kv_heads (8) x p->head_dim (128)
    w->wk_norm = ptr;
    ptr += p->head_dim; // head_dim (128)
    w->rms_att_weight = ptr;
    ptr += p->dim; // dimension (1024)
    w->wo = ptr;
    ptr += (p->n_heads * p->head_dim) * p->dim; // attention heads (16) x head dim (128) * dim
    w->wq = ptr;
    ptr += p->dim * (p->n_heads * p->head_dim);
    w->wq_norm = ptr;
    ptr += p->head_dim; // head_dim (128)
    w->wv = ptr;
    ptr += p->dim * (p->n_kv_heads * p->head_dim); // equal to wk
    w->w2 = ptr;
    ptr += p->hidden_dim * p->dim; // ffn.down 3072 *1024
    w->w3 = ptr;
    ptr += p->dim * p->hidden_dim; // ffn.gate
    w->rms_ffn_weight = ptr;
    ptr += p->dim;                 // ffn.norm
    w->w1 = ptr;
    ptr += p->dim * p->hidden_dim; // ffn.up
}

// ----------------------------------------------------------------------------
// read GGUF
void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights, int* fd, float** data, ssize_t* file_size) {
    FILE* file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = (ssize_t) ftell(file); // get the file size, in bytes
    fclose(file);

    printf("file size is %zd", *file_size);

    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }

    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }

    void* weights_ptr = ((char*)*data) + 5951648; // skip header bytes. header_size = 5951648 TODO
                                                  // gguf total header = file size - (last tensor size + last offset)
    memory_map_weights(weights, config, weights_ptr);
}

void build_transformer(Transformer* t, char* checkpoint_path) {
    // read in the Weights from the GGUF
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    if (t->data && t-> data != MAP_FAILED) {
        munmap()
    }
}
