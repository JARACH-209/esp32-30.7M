// RQ42 quantized forward pass. Faithful port of karpathy's run.c float path
// with group-quantized weights dequantized on the fly. See runq.h.

#include "runq.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// ---- fp16 -> fp32 (scales are stored as IEEE half) -------------------------

static inline float half_to_float(const uint8_t *p) {
  uint16_t h = (uint16_t)(p[0] | (p[1] << 8));
  uint32_t sign = (uint32_t)(h >> 15) & 1u;
  uint32_t exp = (uint32_t)(h >> 10) & 0x1Fu;
  uint32_t man = (uint32_t)h & 0x3FFu;
  uint32_t f;
  if (exp == 0) {
    if (man == 0) { f = sign << 31; }
    else {                                   // subnormal half -> normal float
      int e = -1;
      do { man <<= 1; e++; } while (!(man & 0x400u));
      man &= 0x3FFu;
      f = (sign << 31) | ((uint32_t)(127 - 15 - e) << 23) | (man << 13);
    }
  } else if (exp == 31) {
    f = (sign << 31) | (0xFFu << 23) | (man << 13);
  } else {
    f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float out;
  memcpy(&out, &f, 4);
  return out;
}

#if RQ_KV16
static inline uint16_t float_to_half(float f) {
  uint32_t x; memcpy(&x, &f, 4);
  uint16_t sign = (uint16_t)((x >> 16) & 0x8000u);
  int32_t e = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
  uint32_t m = x & 0x7FFFFFu;
  if (e <= 0) {
    if (e < -10) return sign;
    m |= 0x800000u;
    uint32_t shift = (uint32_t)(14 - e);
    uint16_t half = (uint16_t)(m >> shift);
    if ((m >> (shift - 1)) & 1u) half++;
    return (uint16_t)(sign | half);
  }
  if (e >= 31) return (uint16_t)(sign | 0x7C00u);
  uint16_t out = (uint16_t)(sign | ((uint32_t)e << 10) | (m >> 13));
  if (m & 0x1000u) out++;
  return out;
}
static inline float kv_load(const rq_kv_t *p, int i) {
  uint16_t v = p[i];
  uint8_t b[2] = { (uint8_t)(v & 0xFF), (uint8_t)(v >> 8) };
  return half_to_float(b);
}
#else
static inline float kv_load(const rq_kv_t *p, int i) { return p[i]; }
#endif

size_t rq_row_bytes(int n_in, int bits, int gs) {
  size_t per_group = (bits == 4) ? (size_t)(gs / 2 + 2)
                                 : (size_t)((gs * 3 + 7) / 8 + 2);
  return (size_t)(n_in / gs) * per_group;
}

// ---- kernels ---------------------------------------------------------------

// Dequantize one group of `gs` values scaled by the trailing fp16.
static void dequant_group(const uint8_t *g, int bits, int gs, float *out) {
  if (bits == 4) {
    float scale = half_to_float(g + gs / 2);
    for (int i = 0; i < gs / 2; i++) {
      out[2 * i]     = (float)((int)(g[i] & 0x0F) - 8) * scale;
      out[2 * i + 1] = (float)((int)(g[i] >> 4) - 8) * scale;
    }
  } else {                                    // 3-bit LSB-first bitstream
    int nbytes = (gs * 3 + 7) / 8;
    float scale = half_to_float(g + nbytes);
    uint32_t acc = 0; int nbits = 0, bi = 0;
    for (int i = 0; i < gs; i++) {
      while (nbits < 3) { acc |= (uint32_t)g[bi++] << nbits; nbits += 8; }
      out[i] = (float)((int)(acc & 0x7u) - 4) * scale;
      acc >>= 3; nbits -= 3;
    }
  }
}

void (*rq_matvec_hook)(float *, const float *, const uint8_t *,
                       int, int, int, int) = 0;

// W4A8 fused kernel. The activation vector is quantized to int8 once (one
// scale per group), then every weight row is a pure INTEGER dot product -
// unpack nibble, multiply-accumulate: with one float multiply per group to
// apply both scales. The previous float path decoded every weight to fp32
// through a bounce buffer: ~25 cycles/param measured on the S3. This shape
// runs ~5-7, which finally puts the kernel at the memory floor.
#define RQ_MAX_IN 2048

void rq_matvec_range(float *out, const float *x, const uint8_t *w,
                     int n_in, int row_lo, int row_hi, int bits, int gs) {
  size_t row_bytes = rq_row_bytes(n_in, bits, gs);
  int ng = n_in / gs;
  size_t per_group = row_bytes / (size_t)ng;

  if (bits == 4 && n_in <= RQ_MAX_IN) {
    int8_t xq[RQ_MAX_IN];
    float sx[RQ_MAX_IN / 32];
    for (int k = 0; k < ng; k++) {
      const float *xg = x + k * gs;
      float mx = 0.0f;
      for (int i = 0; i < gs; i++) {
        float a = xg[i] < 0 ? -xg[i] : xg[i];
        if (a > mx) mx = a;
      }
      float s = mx > 0.0f ? mx / 127.0f : 1.0f;
      float inv = 1.0f / s;
      sx[k] = s;
      int8_t *q = xq + k * gs;
      for (int i = 0; i < gs; i++) {
        float v = xg[i] * inv;
        q[i] = (int8_t)(v >= 0.0f ? v + 0.5f : v - 0.5f);
      }
    }
    for (int r = row_lo; r < row_hi; r++) {
      const uint8_t *row = w + (size_t)r * row_bytes;
      float facc = 0.0f;
      for (int k = 0; k < ng; k++) {
        const uint8_t *g = row + (size_t)k * per_group;
        const int8_t *q = xq + k * gs;
        int32_t iacc = 0;
        for (int i = 0; i < gs / 2; i++) {
          int b = g[i];
          iacc += ((b & 0x0F) - 8) * q[2 * i] + ((b >> 4) - 8) * q[2 * i + 1];
        }
        facc += (float)iacc * half_to_float(g + gs / 2) * sx[k];
      }
      out[r] = facc;
    }
    return;
  }

  // Reference float path (Q3 tables, oversized inputs, and the host fallback).
  float dq[512];                               // gs <= 512
  for (int r = row_lo; r < row_hi; r++) {
    const uint8_t *row = w + (size_t)r * row_bytes;
    float acc = 0.0f;
    for (int k = 0; k < ng; k++) {
      dequant_group(row + (size_t)k * per_group, bits, gs, dq);
      const float *xg = x + k * gs;
      for (int i = 0; i < gs; i++) acc += dq[i] * xg[i];
    }
    out[r] = acc;
  }
}

void rq_matvec(float *out, const float *x, const uint8_t *w,
               int n_in, int n_out, int bits, int gs) {
  rq_matvec_range(out, x, w, n_in, 0, n_out, bits, gs);
}

// Every weight matvec in the forward pass goes through here.
static inline void mv(float *out, const float *x, const uint8_t *w,
                      int n_in, int n_out, int bits, int gs) {
  if (rq_matvec_hook) rq_matvec_hook(out, x, w, n_in, n_out, bits, gs);
  else rq_matvec(out, x, w, n_in, n_out, bits, gs);
}

void rq_dequant_row(const RQModel *m, int row, float *out) {
  const uint8_t *r = m->table + (size_t)row * m->table_row_bytes;
  size_t per_group = m->table_row_bytes / (size_t)(m->h.dim / m->h.group);
  for (int k = 0; k < m->h.dim / m->h.group; k++)
    dequant_group(r + (size_t)k * per_group, m->h.table_bits, m->h.group,
                  out + k * m->h.group);
}

static void rmsnorm(float *o, const float *x, const float *w, int n) {
  float ss = 0.0f;
  for (int i = 0; i < n; i++) ss += x[i] * x[i];
  ss = 1.0f / sqrtf(ss / n + 1e-5f);
  for (int i = 0; i < n; i++) o[i] = w[i] * (ss * x[i]);
}

static void softmax_inplace(float *x, int n) {
  float mx = x[0];
  for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
  float sum = 0.0f;
  for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
  for (int i = 0; i < n; i++) x[i] /= sum;
}

// ---- init -------------------------------------------------------------------

int rq_init(RQModel *m, const uint8_t *image, size_t image_len) {
  if (image_len < sizeof(RQHeader)) return -1;
  memcpy(&m->h, image, sizeof(RQHeader));
  if (m->h.magic != RQ42_MAGIC || m->h.version != 1) return -2;
  const RQHeader *h = &m->h;
  if ((size_t)h->core_off + h->core_len > image_len) return -3;

  int L = h->n_layers, dim = h->dim;
  const float *norms = (const float *)(image + h->norms_off);
  m->rms_att = norms;
  m->rms_ffn = norms + (size_t)L * dim;
  m->rms_final = norms + 2 * (size_t)L * dim;
  m->table = image + h->table_off;
  m->core = image + h->core_off;
  m->remap = (const uint16_t *)(image + h->remap_off);

  int kv_dim = (dim / h->n_heads) * h->n_kv_heads;
  m->table_row_bytes = rq_row_bytes(dim, h->table_bits, h->group);
  m->off_wq = 0;
  m->off_wk = m->off_wq + (size_t)dim * rq_row_bytes(dim, h->core_bits, h->group);
  m->off_wv = m->off_wk + (size_t)kv_dim * rq_row_bytes(dim, h->core_bits, h->group);
  m->off_wo = m->off_wv + (size_t)kv_dim * rq_row_bytes(dim, h->core_bits, h->group);
  m->off_w1 = m->off_wo + (size_t)dim * rq_row_bytes(dim, h->core_bits, h->group);
  m->off_w2 = m->off_w1 + (size_t)h->hidden * rq_row_bytes(dim, h->core_bits, h->group);
  m->off_w3 = m->off_w2 + (size_t)dim * rq_row_bytes(h->hidden, h->core_bits, h->group);
  m->core_layer_bytes = m->off_w3 + (size_t)h->hidden * rq_row_bytes(dim, h->core_bits, h->group);

  size_t expect = (size_t)L * m->core_layer_bytes;
  if (expect != h->core_len) return -4;
  if (L > (int)(sizeof(m->layer_ptr) / sizeof(m->layer_ptr[0]))) return -6;
  for (int l = 0; l < L; l++)
    m->layer_ptr[l] = m->core + (size_t)l * m->core_layer_bytes;
  if ((size_t)h->vocab * m->table_row_bytes != h->table_len) return -5;
  return 0;
}

int rq_state_alloc(RQState *s, const RQModel *m, int seq_len) {
  const RQHeader *h = &m->h;
  int dim = h->dim, hid = h->hidden;
  int kv_dim = (dim / h->n_heads) * h->n_kv_heads;
  if (seq_len <= 0 || seq_len > h->seq_len) seq_len = h->seq_len;
  s->seq_len = seq_len;
  s->x = calloc(dim, 4); s->xb = calloc(dim, 4); s->xb2 = calloc(dim, 4);
  s->hb = calloc(hid, 4); s->hb2 = calloc(hid, 4);
  s->q = calloc(dim, 4);
  s->att = calloc((size_t)h->n_heads * seq_len, 4);
  s->logits = calloc(h->vocab, 4);
  s->key_cache = calloc((size_t)h->n_layers * seq_len * kv_dim, sizeof(rq_kv_t));
  s->value_cache = calloc((size_t)h->n_layers * seq_len * kv_dim, sizeof(rq_kv_t));
  return (s->x && s->xb && s->xb2 && s->hb && s->hb2 && s->q && s->att &&
          s->logits && s->key_cache && s->value_cache) ? 0 : -1;
}

// ---- forward -----------------------------------------------------------------

void rq_forward(const RQModel *m, RQState *s, int token, int pos) {
  const RQHeader *h = &m->h;
  int dim = h->dim, hid = h->hidden, heads = h->n_heads;
  int head_size = dim / heads;
  int kv_dim = head_size * h->n_kv_heads;
  int kv_mul = heads / h->n_kv_heads;
  int gs = h->group, cb = h->core_bits;

  rq_dequant_row(m, token, s->x);

  for (int l = 0; l < h->n_layers; l++) {
    const uint8_t *layer = m->layer_ptr[l];
    rmsnorm(s->xb, s->x, m->rms_att + (size_t)l * dim, dim);

    float kbuf[512], vbuf[512];               /* kv_dim <= 512 on this chip */
    float *k = kbuf, *v = vbuf;
    mv(s->q, s->xb, layer + m->off_wq, dim, dim, cb, gs);
    mv(k,    s->xb, layer + m->off_wk, dim, kv_dim, cb, gs);
    mv(v,    s->xb, layer + m->off_wv, dim, kv_dim, cb, gs);

    // RoPE, run.c-style: adjacent pairs, frequency by position within head.
    for (int i = 0; i < dim; i += 2) {
      int hd = i % head_size;
      float freq = 1.0f / powf(10000.0f, (float)hd / (float)head_size);
      float val = pos * freq, fcr = cosf(val), fci = sinf(val);
      int rotn = i < kv_dim ? 2 : 1;
      for (int r = 0; r < rotn; r++) {
        float *vec = r == 0 ? s->q : k;
        float v0 = vec[i], v1 = vec[i + 1];
        vec[i] = v0 * fcr - v1 * fci;
        vec[i + 1] = v0 * fci + v1 * fcr;
      }
    }

    {
      rq_kv_t *kc = s->key_cache + ((size_t)l * s->seq_len + pos) * kv_dim;
      rq_kv_t *vc = s->value_cache + ((size_t)l * s->seq_len + pos) * kv_dim;
#if RQ_KV16
      for (int i = 0; i < kv_dim; i++) { kc[i] = float_to_half(kbuf[i]); vc[i] = float_to_half(vbuf[i]); }
#else
      memcpy(kc, kbuf, (size_t)kv_dim * 4); memcpy(vc, vbuf, (size_t)kv_dim * 4);
#endif
    }

    for (int hh = 0; hh < heads; hh++) {
      const float *qh = s->q + (size_t)hh * head_size;
      float *att = s->att + (size_t)hh * s->seq_len;
      for (int t = 0; t <= pos; t++) {
        const rq_kv_t *kt = s->key_cache +
            ((size_t)l * s->seq_len + t) * kv_dim + (hh / kv_mul) * head_size;
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) score += qh[i] * kv_load(kt, i);
        att[t] = score / sqrtf((float)head_size);
      }
      softmax_inplace(att, pos + 1);
      float *out = s->xb + (size_t)hh * head_size;
      memset(out, 0, (size_t)head_size * 4);
      for (int t = 0; t <= pos; t++) {
        const rq_kv_t *vt = s->value_cache +
            ((size_t)l * s->seq_len + t) * kv_dim + (hh / kv_mul) * head_size;
        float a = att[t];
        for (int i = 0; i < head_size; i++) out[i] += a * kv_load(vt, i);
      }
    }

    mv(s->xb2, s->xb, layer + m->off_wo, dim, dim, cb, gs);
    for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];

    rmsnorm(s->xb, s->x, m->rms_ffn + (size_t)l * dim, dim);
    mv(s->hb, s->xb, layer + m->off_w1, dim, hid, cb, gs);
    mv(s->hb2, s->xb, layer + m->off_w3, dim, hid, cb, gs);
    for (int i = 0; i < hid; i++) {
      float val = s->hb[i];
      val *= 1.0f / (1.0f + expf(-val));        // SiLU
      s->hb[i] = val * s->hb2[i];
    }
    mv(s->xb2, s->hb, layer + m->off_w2, hid, dim, cb, gs);
    for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];
  }

  rmsnorm(s->xb, s->x, m->rms_final, dim);
  // Tied classifier: logits over the KEPT vocabulary from the same table.
  mv(s->logits, s->xb, m->table, dim, h->vocab,
     h->table_bits, h->group);
}
