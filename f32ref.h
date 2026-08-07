/* fp32 reference for llama2.c legacy checkpoints: ported from karpathy's
 * run.c (MIT). Shared by verify.c (the quantization gate) and gen.c (the
 * corpus generator). All functions static: include from exactly one TU each.
 */
#ifndef F32REF_H
#define F32REF_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- fp32 reference (run.c port, abbreviated) --------------------------- */

typedef struct {
  int dim, hidden, n_layers, n_heads, n_kv_heads, vocab, seq_len, tied;
  float *tok_emb, *rms_att, *wq, *wk, *wv, *wo, *rms_ffn, *w1, *w2, *w3,
        *rms_final, *wcls;
} F32Model;

typedef struct {
  float *x, *xb, *xb2, *hb, *hb2, *q, *att, *logits, *kc, *vc;
} F32State;

static float *fptr(float **cur, size_t n) { float *p = *cur; *cur += n; return p; }

static int f32_load(F32Model *m, const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) return -1;
  int hdr[7];
  if (fread(hdr, 4, 7, f) != 7) return -2;
  m->dim = hdr[0]; m->hidden = hdr[1]; m->n_layers = hdr[2];
  m->n_heads = hdr[3]; m->n_kv_heads = hdr[4];
  m->tied = hdr[5] > 0; m->vocab = abs(hdr[5]); m->seq_len = hdr[6];
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 28, SEEK_SET);
  float *data = malloc(sz - 28);
  if (fread(data, 1, sz - 28, f) != (size_t)(sz - 28)) return -3;
  fclose(f);

  int hs = m->dim / m->n_heads, kvd = hs * m->n_kv_heads;
  float *cur = data;
  m->tok_emb  = fptr(&cur, (size_t)m->vocab * m->dim);
  m->rms_att  = fptr(&cur, (size_t)m->n_layers * m->dim);
  m->wq       = fptr(&cur, (size_t)m->n_layers * m->dim * m->dim);
  m->wk       = fptr(&cur, (size_t)m->n_layers * kvd * m->dim);
  m->wv       = fptr(&cur, (size_t)m->n_layers * kvd * m->dim);
  m->wo       = fptr(&cur, (size_t)m->n_layers * m->dim * m->dim);
  m->rms_ffn  = fptr(&cur, (size_t)m->n_layers * m->dim);
  m->w1       = fptr(&cur, (size_t)m->n_layers * m->hidden * m->dim);
  m->w2       = fptr(&cur, (size_t)m->n_layers * m->dim * m->hidden);
  m->w3       = fptr(&cur, (size_t)m->n_layers * m->hidden * m->dim);
  m->rms_final = fptr(&cur, m->dim);
  cur += (size_t)m->seq_len * hs;               /* legacy freq_cis re+im */
  m->wcls = m->tied ? m->tok_emb : fptr(&cur, (size_t)m->vocab * m->dim);
  return 0;
}

static void f32_state(F32State *s, const F32Model *m) {
  int kvd = (m->dim / m->n_heads) * m->n_kv_heads;
  s->x = calloc(m->dim, 4); s->xb = calloc(m->dim, 4); s->xb2 = calloc(m->dim, 4);
  s->hb = calloc(m->hidden, 4); s->hb2 = calloc(m->hidden, 4);
  s->q = calloc(m->dim, 4);
  s->att = calloc((size_t)m->n_heads * m->seq_len, 4);
  s->logits = calloc(m->vocab, 4);
  s->kc = calloc((size_t)m->n_layers * m->seq_len * kvd, 4);
  s->vc = calloc((size_t)m->n_layers * m->seq_len * kvd, 4);
}

static void f32_rmsnorm(float *o, const float *x, const float *w, int n) {
  float ss = 0;
  for (int i = 0; i < n; i++) ss += x[i] * x[i];
  ss = 1.0f / sqrtf(ss / n + 1e-5f);
  for (int i = 0; i < n; i++) o[i] = w[i] * (ss * x[i]);
}
static void f32_softmax(float *x, int n) {
  float mx = x[0];
  for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
  float s = 0;
  for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
  for (int i = 0; i < n; i++) x[i] /= s;
}
static void f32_matvec(float *out, const float *x, const float *w, int n, int d) {
  for (int r = 0; r < d; r++) {
    float acc = 0;
    const float *row = w + (size_t)r * n;
    for (int i = 0; i < n; i++) acc += row[i] * x[i];
    out[r] = acc;
  }
}

static void f32_forward(const F32Model *m, F32State *s, int token, int pos) {
  int dim = m->dim, hid = m->hidden, heads = m->n_heads;
  int hs = dim / heads, kvd = hs * m->n_kv_heads, kvm = heads / m->n_kv_heads;
  memcpy(s->x, m->tok_emb + (size_t)token * dim, dim * 4);

  for (int l = 0; l < m->n_layers; l++) {
    f32_rmsnorm(s->xb, s->x, m->rms_att + (size_t)l * dim, dim);
    float *k = s->kc + ((size_t)l * m->seq_len + pos) * kvd;
    float *v = s->vc + ((size_t)l * m->seq_len + pos) * kvd;
    f32_matvec(s->q, s->xb, m->wq + (size_t)l * dim * dim, dim, dim);
    f32_matvec(k, s->xb, m->wk + (size_t)l * kvd * dim, dim, kvd);
    f32_matvec(v, s->xb, m->wv + (size_t)l * kvd * dim, dim, kvd);

    for (int i = 0; i < dim; i += 2) {
      int hd = i % hs;
      float freq = 1.0f / powf(10000.0f, (float)hd / (float)hs);
      float val = pos * freq, fcr = cosf(val), fci = sinf(val);
      int rotn = i < kvd ? 2 : 1;
      for (int r = 0; r < rotn; r++) {
        float *vec = r == 0 ? s->q : k;
        float v0 = vec[i], v1 = vec[i + 1];
        vec[i] = v0 * fcr - v1 * fci;
        vec[i + 1] = v0 * fci + v1 * fcr;
      }
    }
    for (int hh = 0; hh < heads; hh++) {
      const float *qh = s->q + (size_t)hh * hs;
      float *att = s->att + (size_t)hh * m->seq_len;
      for (int t = 0; t <= pos; t++) {
        const float *kt = s->kc + ((size_t)l * m->seq_len + t) * kvd + (hh / kvm) * hs;
        float sc = 0;
        for (int i = 0; i < hs; i++) sc += qh[i] * kt[i];
        att[t] = sc / sqrtf((float)hs);
      }
      f32_softmax(att, pos + 1);
      float *out = s->xb + (size_t)hh * hs;
      memset(out, 0, (size_t)hs * 4);
      for (int t = 0; t <= pos; t++) {
        const float *vt = s->vc + ((size_t)l * m->seq_len + t) * kvd + (hh / kvm) * hs;
        for (int i = 0; i < hs; i++) out[i] += att[t] * vt[i];
      }
    }
    f32_matvec(s->xb2, s->xb, m->wo + (size_t)l * dim * dim, dim, dim);
    for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];
    f32_rmsnorm(s->xb, s->x, m->rms_ffn + (size_t)l * dim, dim);
    f32_matvec(s->hb, s->xb, m->w1 + (size_t)l * hid * dim, dim, hid);
    f32_matvec(s->hb2, s->xb, m->w3 + (size_t)l * hid * dim, dim, hid);
    for (int i = 0; i < hid; i++) {
      float val = s->hb[i];
      val *= 1.0f / (1.0f + expf(-val));
      s->hb[i] = val * s->hb2[i];
    }
    f32_matvec(s->xb2, s->hb, m->w2 + (size_t)l * dim * hid, hid, dim);
    for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];
  }
  f32_rmsnorm(s->xb, s->x, m->rms_final, dim);
  f32_matvec(s->logits, s->xb, m->wcls, dim, m->vocab);
}


#endif
