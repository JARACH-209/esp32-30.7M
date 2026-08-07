/* Host gate for the RQ42 pack: the quantized runtime vs karpathy's fp32 math,
 * token by token, on the actual checkpoint.
 *
 * Loads BOTH the original fp32 .bin (reference forward pass ported from run.c)
 * and the quantized pack (the exact runq.c the device runs). Then:
 *
 *   1. teacher-forces the same token sequence through both and reports top-1
 *      agreement + logit error, restricted to the kept vocabulary;
 *   2. greedy-generates from the same prompt on both and prints both stories;
 *   3. for trimmed packs, reports corpus coverage: how often fp32's TRUE
 *      argmax fell outside the kept set (the honest cost of the trim).
 *
 *   cc -O2 -Wall -Wextra -o verify verify.c runq.c -lm
 *   ./verify model.bin pack.bin tokenizer.bin [n_steps]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "runq.h"

#include "f32ref.h"

/* ---- tokenizer (decode only) --------------------------------------------- */

typedef struct { char **str; int n; } Tok;

static int tok_load(Tok *t, const char *path, int vocab) {
  FILE *f = fopen(path, "rb");
  if (!f) return -1;
  int maxlen;
  if (fread(&maxlen, 4, 1, f) != 1) return -2;
  t->n = vocab;
  t->str = calloc(vocab, sizeof(char *));
  for (int i = 0; i < vocab; i++) {
    float score; int len;
    if (fread(&score, 4, 1, f) != 1 || fread(&len, 4, 1, f) != 1) return -3;
    t->str[i] = calloc(len + 1, 1);
    if (fread(t->str[i], 1, len, f) != (size_t)len) return -4;
  }
  fclose(f);
  return 0;
}

static void tok_print(const Tok *t, int id, int prev) {
  if (id < 0 || id >= t->n) { printf("<?>"); return; }
  const char *s = t->str[id];
  if (prev == 1 && s[0] == ' ') s++;                 /* BOS strip, run.c-style */
  unsigned char b;
  if (sscanf(s, "<0x%02hhX>", &b) == 1) putchar(b);  /* byte-fallback token  */
  else fputs(s, stdout);
}

/* ---- gate ----------------------------------------------------------------- */

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s model.bin pack.bin tokenizer.bin [steps]\n", argv[0]);
    return 2;
  }
  int steps = argc > 4 ? atoi(argv[4]) : 120;

  F32Model fm;
  if (f32_load(&fm, argv[1])) { fprintf(stderr, "bad fp32 model\n"); return 1; }
  F32State fs;
  f32_state(&fs, &fm);

  FILE *pf = fopen(argv[2], "rb");
  if (!pf) { perror(argv[2]); return 1; }
  fseek(pf, 0, SEEK_END);
  long psz = ftell(pf);
  fseek(pf, 0, SEEK_SET);
  uint8_t *img = malloc(psz);
  if (fread(img, 1, psz, pf) != (size_t)psz) { perror("read"); return 1; }
  fclose(pf);

  RQModel qm;
  int rc = rq_init(&qm, img, psz);
  if (rc) { fprintf(stderr, "rq_init failed: %d\n", rc); return 1; }
  RQState qs;
  if (rq_state_alloc(&qs, &qm, 0)) { fprintf(stderr, "state alloc\n"); return 1; }

  Tok tok;
  if (tok_load(&tok, argv[3], fm.vocab)) { fprintf(stderr, "bad tokenizer\n"); return 1; }

  printf("fp32 : vocab %d, dim %d, %d layers  (%s)\n", fm.vocab, fm.dim,
         fm.n_layers, fm.tied ? "tied" : "untied");
  printf("pack : vocab %d of %d kept, core Q%d, table Q%d, group %d\n\n",
         qm.h.vocab, qm.h.vocab_orig, qm.h.core_bits, qm.h.table_bits, qm.h.group);

  /* original id -> kept row, or -1 */
  int *old2new = malloc((size_t)fm.vocab * sizeof(int));
  for (int i = 0; i < fm.vocab; i++) old2new[i] = -1;
  for (int i = 0; i < qm.h.vocab; i++) old2new[qm.remap[i]] = i;

  /* -- pass 1: fp32 greedy rollout = the forcing sequence ------------------ */
  int *seq = malloc((size_t)(steps + 1) * sizeof(int));
  seq[0] = 1;                                        /* BOS */
  for (int p = 0; p < steps; p++) {
    f32_forward(&fm, &fs, seq[p], p);
    int best = 0;
    for (int v = 1; v < fm.vocab; v++)
      if (fs.logits[v] > fs.logits[best]) best = v;
    seq[p + 1] = best;
  }

  /* -- pass 2: teacher-force both, compare ---------------------------------- */
  memset(fs.kc, 0, 1);                               /* caches restart at pos 0 */
  int agree = 0, compared = 0, dropped = 0;
  double logit_err = 0, ref_gap = 0;
  for (int p = 0; p < steps; p++) {
    int tok_old = seq[p];
    int tok_new = old2new[tok_old];
    if (tok_new < 0) { dropped++; break; }           /* forcing left kept set  */
    f32_forward(&fm, &fs, tok_old, p);
    rq_forward(&qm, &qs, tok_new, p);

    /* fp32 argmax restricted to kept ids vs quant argmax */
    int fbest = -1; float fbv = -1e30f;
    for (int i = 0; i < qm.h.vocab; i++) {
      float lv = fs.logits[qm.remap[i]];
      if (lv > fbv) { fbv = lv; fbest = i; }
    }
    int qbest = 0;
    for (int i = 1; i < qm.h.vocab; i++)
      if (qs.logits[i] > qs.logits[qbest]) qbest = i;

    /* also: did the trim hide fp32's TRUE argmax? */
    int tbest = 0;
    for (int v = 1; v < fm.vocab; v++)
      if (fs.logits[v] > fs.logits[tbest]) tbest = v;
    if (old2new[tbest] < 0) ref_gap += 1.0;

    if (fbest == qbest) agree++;
    logit_err += fabs((double)fbv - (double)qs.logits[qbest]);
    compared++;
  }

  printf("teacher-forced %d positions:\n", compared);
  printf("  top-1 agreement (kept-set)   %.1f%%  (%d/%d)\n",
         100.0 * agree / (compared ? compared : 1), agree, compared);
  printf("  mean |top-1 logit delta|     %.4f\n",
         logit_err / (compared ? compared : 1));
  printf("  fp32 true argmax outside kept set: %.1f%% of positions\n",
         100.0 * ref_gap / (compared ? compared : 1));
  if (dropped) printf("  NOTE: forcing sequence left the kept set after %d steps\n",
                      compared);

  /* -- pass 3: both stories, greedy ------------------------------------------ */
  printf("\n--- fp32 story ---\n");
  int t = 1;
  for (int p = 0; p < steps; p++) {
    f32_forward(&fm, &fs, t, p);
    int best = 0;
    for (int v = 1; v < fm.vocab; v++)
      if (fs.logits[v] > fs.logits[best]) best = v;
    tok_print(&tok, best, t); t = best;
    if (t == 2 || t == 1) break;                     /* EOS/BOS */
  }
  printf("\n\n--- quantized story ---\n");
  int tq = old2new[1];
  for (int p = 0; p < steps; p++) {
    rq_forward(&qm, &qs, tq, p);
    int best = 0;
    for (int i = 1; i < qm.h.vocab; i++)
      if (qs.logits[i] > qs.logits[best]) best = i;
    tok_print(&tok, qm.remap[best], qm.remap[tq]); tq = best;
    if (qm.remap[tq] == 2 || qm.remap[tq] == 1) break;
  }
  printf("\n\nverdict: %s\n",
         (compared && 100.0 * agree / compared >= 85.0)
             ? "PASS (>=85% agreement)" : "REVIEW (below 85% agreement)");
  return 0;
}
