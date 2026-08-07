/* Corpus generator for vocabulary ranking. Loads the fp32 checkpoint, samples
 * with temperature, prints one token id per line. The frequency census over
 * this output decides which table rows the trimmed pack keeps: the model
 * votes on its own vocabulary.
 *
 *   cc -O2 -o gen gen.c -lm
 *   ./gen model.bin 30000 1.0 1 > ids_1.txt     # steps, temperature, seed
 *
 * Several seeds catenated give a stable census; 100-200k tokens is plenty.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* fp32 reference model: same layout as verify.c, kept standalone so this tool
 * builds with no other files. */
#include "f32ref.h"

static unsigned long long rng_state;
static float rng_f32(void) {                    /* xorshift, run.c-style */
  rng_state ^= rng_state >> 12;
  rng_state ^= rng_state << 25;
  rng_state ^= rng_state >> 27;
  return (float)((rng_state * 0x2545F4914F6CDD1Dull) >> 40) / 16777216.0f;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s model.bin steps temperature [seed]\n", argv[0]);
    return 2;
  }
  int steps = atoi(argv[2]);
  float temp = (float)atof(argv[3]);
  rng_state = argc > 4 ? strtoull(argv[4], NULL, 10) : 1234567ull;

  F32Model m;
  if (f32_load(&m, argv[1])) { fprintf(stderr, "bad model\n"); return 1; }
  F32State s;
  f32_state(&s, &m);

  int tok = 1;                                   /* BOS */
  printf("%d\n", tok);
  for (int pos = 0; pos < steps; pos++) {
    if (pos >= m.seq_len) break;
    f32_forward(&m, &s, tok, pos);
    int next;
    if (temp <= 0.0f) {
      next = 0;
      for (int v = 1; v < m.vocab; v++)
        if (s.logits[v] > s.logits[next]) next = v;
    } else {
      for (int v = 0; v < m.vocab; v++) s.logits[v] /= temp;
      /* softmax + multinomial */
      float mx = s.logits[0];
      for (int v = 1; v < m.vocab; v++) if (s.logits[v] > mx) mx = s.logits[v];
      float sum = 0;
      for (int v = 0; v < m.vocab; v++) { s.logits[v] = expf(s.logits[v] - mx); sum += s.logits[v]; }
      float r = rng_f32() * sum, acc = 0;
      next = m.vocab - 1;
      for (int v = 0; v < m.vocab; v++) {
        acc += s.logits[v];
        if (r < acc) { next = v; break; }
      }
    }
    printf("%d\n", next);
    tok = next;
    if (tok == 1) {                              /* BOS again = new story */
      /* keep going: stories chain, positions reset would need cache reset;
         simplest correct behaviour is to stop at seq_len which the loop does */
    }
  }
  return 0;
}
