// RQ42 quantized runtime: shared between the host verify gate and the ESP32
// sketch. Plain C99, no dependencies. The backend supplies three base pointers
// (norms, table, core); host points them into a malloc'd file image, device
// points them at SRAM copies / PSRAM stages / memory-mapped flash. The forward
// pass is identical on both, which is the entire point: every numerical
// decision is gated on host before it touches the board.

#ifndef RUNQ_H
#define RUNQ_H

#include <stdint.h>
#include <stddef.h>

#define RQ42_MAGIC 0x32345152u

typedef struct {
  uint32_t magic, version;
  int32_t dim, hidden, n_layers, n_heads, n_kv_heads;
  int32_t vocab;          // kept vocabulary (table rows)
  int32_t vocab_orig;     // original checkpoint vocabulary
  int32_t seq_len;
  int32_t group, core_bits, table_bits;
  int32_t hidden_true;    // pre-padding FFN dim, for honest param accounting
  uint32_t flags;         // bit0: tied classifier
  uint32_t norms_off, norms_len;
  uint32_t table_off, table_len;
  uint32_t core_off, core_len;
  uint32_t remap_off, remap_len;
  uint32_t crc32;
} RQHeader;

typedef struct {
  RQHeader h;
  const float *rms_att;    // [n_layers][dim]
  const float *rms_ffn;    // [n_layers][dim]
  const float *rms_final;  // [dim]
  const uint8_t *table;    // kept-vocab rows, quantized
  const uint8_t *core;     // per layer: wq wk wv wo w1 w2 w3
  const uint16_t *remap;   // kept row -> original token id (host-side use)
  // strides, precomputed by rq_init
  size_t table_row_bytes;
  size_t core_layer_bytes;
  size_t off_wq, off_wk, off_wv, off_wo, off_w1, off_w2, off_w3;
  // Per-layer weight base, defaulted by rq_init to core + l*core_layer_bytes.
  // The device overrides entries (and m->table) after copying sections into
  // PSRAM: the forward pass reads only through these, so staging is invisible
  // to the math. 16 covers any model this chip could hold.
  const uint8_t *layer_ptr[16];
} RQModel;

// KV cache precision: build with -DRQ_KV16=1 to store K/V as IEEE half,
// halving cache memory (the device ships this way; gate on host with the same
// flag). Default fp32 keeps the pure-quantization comparison clean.
#ifndef RQ_KV16
#define RQ_KV16 0
#endif
#if RQ_KV16
typedef uint16_t rq_kv_t;
#else
typedef float rq_kv_t;
#endif

typedef struct {
  float *x, *xb, *xb2, *hb, *hb2, *q, *att, *logits;
  rq_kv_t *key_cache, *value_cache; // [n_layers][seq][kv_dim]
  int seq_len;                      // KV budget actually allocated
} RQState;

// Parse header + set section pointers from a contiguous image. Returns 0 on
// success. `image` must stay alive for the model's lifetime.
int rq_init(RQModel *m, const uint8_t *image, size_t image_len);

// Allocate state buffers with malloc (host): device allocates its own and
// fills the struct manually to control memory placement.
int rq_state_alloc(RQState *s, const RQModel *m, int seq_len);

// One transformer step: token is a KEPT-vocab row index. Logits (length
// h.vocab) land in s->logits.
void rq_forward(const RQModel *m, RQState *s, int token, int pos);

// Dequantize one table row into out[dim] (the embedding lookup).
void rq_dequant_row(const RQModel *m, int row, float *out);

// Group-quantized matvec: out[r] = sum_i w[r][i] * x[i], w quantized with
// `bits` in groups of `gs`. Exposed for the device to swap in a SIMD version.
void rq_matvec(float *out, const float *x, const uint8_t *w,
               int n_in, int n_out, int bits, int gs);

// Row-range variant, for splitting one matvec across cores: computes
// out[row_lo..row_hi) only. rq_matvec == range over [0, n_out).
void rq_matvec_range(float *out, const float *x, const uint8_t *w,
                     int n_in, int row_lo, int row_hi, int bits, int gs);

// Optional override: when non-NULL, rq_forward routes EVERY weight matvec
// through this instead of rq_matvec. The device points it at a dual-core
// splitter; host verification leaves it NULL, so the gated math is identical.
extern void (*rq_matvec_hook)(float *out, const float *x, const uint8_t *w,
                              int n_in, int n_out, int bits, int gs);

size_t rq_row_bytes(int n_in, int bits, int gs);

#endif
