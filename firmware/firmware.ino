// stories42M on an ESP32-S3: the dense-record sketch.
//
// Every stored parameter participates in a matmul on every generated token:
// the Q4 core streams from memory-mapped flash, the tied table (staged to
// PSRAM at boot, with as many core layers as fit after it) produces logits
// over the full kept vocabulary. No lookup-table shortcuts, no per-layer
// embeddings: the whole pack is read, multiplied, every token.
//
// Build and flash with scripts in record42/: the flash script regenerates
// generated/vocab.h from the pack's own keep list and cross-checks row
// counts, so firmware and weights cannot drift apart.
//
// Serial 115200. Stories generate forever; RESET is never required.

#include <string.h>
#include "esp_partition.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_system.h"
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

extern "C" {
#include "runq.h"
}
#include "generated/vocab.h"

// ---- knobs -------------------------------------------------------------------
#ifndef USE_DISPLAY
#define USE_DISPLAY 1
#endif
#define SEQ_LEN      192       // KV budget: 8L*192*512*2 * 2B (fp16) = 3.0 MB
#define TEMPERATURE  0.85f
#define STORY_PAUSE_MS 4000
#define GEN_TASK_STACK (24 * 1024)
#define PSRAM_STAGE_RESERVE (512 * 1024)   // headroom left after staging

#if !RQ_KV16
#error "build with -DRQ_KV16=1 (the flash script passes it); host-gate the same way"
#endif

// ---- display (optional, SSD1306 128x64 on the proven wiring) ------------------
#if USE_DISPLAY
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#define OLED_SDA 18
#define OLED_SCL 17
#define OLED_ADDR 0x3C
static Adafruit_SSD1306 oled(128, 64, &Wire, -1);
static bool oled_ok = false;
static int oled_col = 0, oled_row = 0;
static char wordbuf[48];
static int wordlen = 0;

static void oled_begin() {
  Wire.begin(OLED_SDA, OLED_SCL);
  Wire.setClock(400000);
  Wire.beginTransmission(OLED_ADDR);
  if (Wire.endTransmission() != 0) return;
  if (!oled.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) return;
  oled.clearDisplay(); oled.setTextSize(1);
  oled.setTextColor(SSD1306_WHITE); oled.setTextWrap(false);
  oled.display();
  oled_ok = true;
}
static void oled_home() { oled.clearDisplay(); oled_col = 0; oled_row = 0; }
static void oled_word_flush() {
  if (!oled_ok || wordlen == 0) { wordlen = 0; return; }
  wordbuf[wordlen] = 0;
  if (oled_col + wordlen > 21) { oled_col = 0; oled_row++; }
  if (oled_row > 7) { oled_home(); }
  oled.setCursor(oled_col * 6, oled_row * 8);
  oled.print(wordbuf);
  oled.display();
  oled_col += wordlen + 1;
  wordlen = 0;
}
static void oled_feed(char c) {
  if (!oled_ok) return;
  if (c == ' ' || c == '\n') oled_word_flush();
  else if (wordlen < (int)sizeof(wordbuf) - 1 && (unsigned char)c >= 0x20)
    wordbuf[wordlen++] = c;
}
static void oled_stats(float tps, int stored_k) {
  if (!oled_ok) return;
  oled_home();
  oled.setCursor(0, 16); oled.print("stories42M  DENSE");
  oled.setCursor(0, 28); oled.printf("%d.%02dM params", stored_k / 1000,
                                     (stored_k % 1000) / 10);
  oled.setCursor(0, 40); oled.printf("%.2f tok/s", tps);
  oled.display();
}
#endif

// ---- dual-core matvec ------------------------------------------------------------
// Barista's pattern: one worker pinned to the other core computes rows
// [0, split) of every large matvec while this core does [split, n_out).
// Both halves write disjoint ranges of out, so no lock is needed: just the
// two notifications. Small matvecs stay single-core: the notify round-trip
// costs more than 64 rows of work.
static TaskHandle_t worker_h = NULL, main_h = NULL;
static struct {
  float *out; const float *x; const uint8_t *w;
  int n_in, split, bits, gs;
} job;

static void worker_main(void *) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    rq_matvec_range(job.out, job.x, job.w, job.n_in, 0, job.split,
                    job.bits, job.gs);
    xTaskNotifyGive(main_h);
  }
}

static void mv_par(float *out, const float *x, const uint8_t *w,
                   int n_in, int n_out, int bits, int gs) {
  if (!worker_h || n_out < 64) {
    rq_matvec(out, x, w, n_in, n_out, bits, gs);
    return;
  }
  job.out = out; job.x = x; job.w = w;
  job.n_in = n_in; job.split = n_out / 2; job.bits = bits; job.gs = gs;
  xTaskNotifyGive(worker_h);
  rq_matvec_range(out, x, w, n_in, job.split, n_out, bits, gs);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
}

// ---- model globals -------------------------------------------------------------
static RQModel M;
static RQState S;
static const uint8_t *g_image = NULL;
static size_t g_pack_bytes = 0;
static int g_bos_row = -1, g_eos_row = -1;
static long g_stored_params = 0;
static int g_staged_layers = 0;

// xorshift sampler (gen.c's)
static unsigned long long rng;
static float rng_f32() {
  rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
  return (float)((rng * 0x2545F4914F6CDD1Dull) >> 40) / 16777216.0f;
}

static int sample_next() {
  int V = M.h.vocab;
  float *lg = S.logits;
  if (TEMPERATURE <= 0.0f) {
    int best = 0;
    for (int v = 1; v < V; v++) if (lg[v] > lg[best]) best = v;
    return best;
  }
  float mx = lg[0];
  for (int v = 1; v < V; v++) if (lg[v] > mx) mx = lg[v];
  float sum = 0;
  for (int v = 0; v < V; v++) { lg[v] = expf((lg[v] - mx) / TEMPERATURE); sum += lg[v]; }
  float r = rng_f32() * sum, acc = 0;
  for (int v = 0; v < V; v++) { acc += lg[v]; if (r < acc) return v; }
  return V - 1;
}

static void emit_row(int row, int prev_row) {
  uint32_t a = VOCAB_OFF[row], b = VOCAB_OFF[row + 1];
  const uint8_t *p = VOCAB_BLOB + a;
  uint32_t n = b - a;
  if (prev_row == g_bos_row && n && p[0] == ' ') { p++; n--; }   // run.c BOS strip
  for (uint32_t i = 0; i < n; i++) {
    Serial.write(p[i]);
#if USE_DISPLAY
    oled_feed((char)p[i]);
#endif
  }
}

// ---- device-placed state alloc ---------------------------------------------------
static void *ps_alloc(size_t n) { return heap_caps_malloc(n, MALLOC_CAP_SPIRAM); }
static void *in_alloc(size_t n) { return heap_caps_malloc(n, MALLOC_CAP_INTERNAL); }

static bool state_alloc_device() {
  const RQHeader *h = &M.h;
  int dim = h->dim, hid = h->hidden;
  int kv_dim = (dim / h->n_heads) * h->n_kv_heads;
  S.seq_len = SEQ_LEN;
  // hot, tiny -> internal SRAM
  S.x  = (float *)in_alloc(dim * 4);  S.xb  = (float *)in_alloc(dim * 4);
  S.xb2 = (float *)in_alloc(dim * 4); S.q   = (float *)in_alloc(dim * 4);
  S.hb = (float *)in_alloc(hid * 4);  S.hb2 = (float *)in_alloc(hid * 4);
  S.att = (float *)in_alloc((size_t)h->n_heads * SEQ_LEN * 4);
  // larger -> PSRAM
  S.logits = (float *)ps_alloc((size_t)h->vocab * 4);
  S.key_cache   = (rq_kv_t *)ps_alloc((size_t)h->n_layers * SEQ_LEN * kv_dim * sizeof(rq_kv_t));
  S.value_cache = (rq_kv_t *)ps_alloc((size_t)h->n_layers * SEQ_LEN * kv_dim * sizeof(rq_kv_t));
  return S.x && S.xb && S.xb2 && S.q && S.hb && S.hb2 && S.att &&
         S.logits && S.key_cache && S.value_cache;
}

// Copy the table then as many core layers as fit into PSRAM; repoint the model.
static void stage_to_psram() {
  uint8_t *t = (uint8_t *)ps_alloc(M.h.table_len);
  if (t) {
    memcpy(t, M.table, M.h.table_len);
    M.table = t;
    Serial.printf("staged table -> PSRAM (%.2f MB)\n", M.h.table_len / 1048576.0);
  }
  for (int l = 0; l < M.h.n_layers; l++) {
    if (heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM) <
        M.core_layer_bytes + PSRAM_STAGE_RESERVE) break;
    uint8_t *b = (uint8_t *)ps_alloc(M.core_layer_bytes);
    if (!b) break;
    memcpy(b, M.layer_ptr[l], M.core_layer_bytes);
    M.layer_ptr[l] = b;
    g_staged_layers++;
  }
  Serial.printf("staged %d/%d core layers -> PSRAM | psram free %.2f MB\n",
                g_staged_layers, (int)M.h.n_layers,
                heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1048576.0);
}

// ---- generation task ---------------------------------------------------------------
static void gen_task(void *arg) {
  (void)arg;
  main_h = xTaskGetCurrentTaskHandle();
  for (;;) {
    int pos = 0, tok = g_bos_row, prev = g_bos_row, produced = 0;
    Serial.print("\n>>> ");
#if USE_DISPLAY
    if (oled_ok) oled_home();
#endif
    int64_t t0 = esp_timer_get_time();
    int64_t compute_us = 0;
    while (pos < SEQ_LEN) {
      int64_t c0 = esp_timer_get_time();
      rq_forward(&M, &S, tok, pos);
      compute_us += esp_timer_get_time() - c0;
      int next = sample_next();
      pos++; produced++;
      if (next == g_eos_row || next == g_bos_row) break;
      emit_row(next, prev);
      prev = tok = next;
      if ((produced & 7) == 0) vTaskDelay(1);            // feed the WDT
      if ((produced & 31) == 0) {
        float tps = produced * 1e6f / (esp_timer_get_time() - t0);
        Serial.printf("  [%.2f tok/s]", tps);
      }
    }
#if USE_DISPLAY
    oled_word_flush();
#endif
    double secs = (esp_timer_get_time() - t0) / 1e6;
    float tps = produced / (float)secs;
    Serial.printf("\n--- %d tokens | %.1f s | %.2f tok/s "
                  "(compute %.0f ms/tok) | %.2fM params dense ---\n",
                  produced, secs, tps, compute_us / 1000.0 / produced,
                  g_stored_params / 1e6);
#if USE_DISPLAY
    oled_stats(tps, (int)(g_stored_params / 1000));
#endif
    vTaskDelay(pdMS_TO_TICKS(STORY_PAUSE_MS));
  }
}

// ---- boot -----------------------------------------------------------------------------
void setup() {
  Serial.setTxBufferSize(4096);
  Serial.begin(115200);
  delay(2000);
  Serial.println("\n=== stories42M on ESP32-S3 :: dense record run ===");

#if USE_DISPLAY
  oled_begin();
  if (!oled_ok) Serial.println("no display at 0x3C; serial only");
#endif

  const esp_partition_t *part = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "model");
  if (!part) { Serial.println("FATAL: no model partition"); return; }
  const void *base; esp_partition_mmap_handle_t h;
  if (esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA,
                         &base, &h) != ESP_OK) {
    Serial.println("FATAL: mmap failed"); return;
  }
  g_image = (const uint8_t *)base;

  int rc = rq_init(&M, g_image, part->size);
  if (rc) { Serial.printf("FATAL: rq_init %d (blank partition? run flash_record.sh)\n", rc); return; }

  g_pack_bytes = M.h.remap_off + M.h.remap_len;
  // Provenance: FNV-1a over the pack, must match what the flash script printed.
  uint32_t fp = 2166136261u;
  for (size_t i = 0; i < g_pack_bytes; i++) { fp ^= g_image[i]; fp *= 16777619u; }

  // Stored params: real core (padding excluded) + kept table rows.
  {
    const RQHeader *hh = &M.h;
    int hs = hh->dim / hh->n_heads, kvd = hs * hh->n_kv_heads;
    long per_layer = 2L * hh->dim * hh->dim + 2L * kvd * hh->dim;   // wq wo wk wv
    // hidden_true excludes the inert zero pad lanes, so the banner reports
    // trained parameters, not storage padding.
    long hid = hh->hidden_true > 0 ? hh->hidden_true : hh->hidden;
    per_layer += 3L * hid * hh->dim;
    g_stored_params = (long)hh->n_layers * per_layer + (long)hh->vocab * hh->dim;
  }

  Serial.printf("model: dim=%d hid=%d L=%d H=%d kv=%d V=%d (of %d) seq=%d\n",
                M.h.dim, M.h.hidden, M.h.n_layers, M.h.n_heads, M.h.n_kv_heads,
                M.h.vocab, M.h.vocab_orig, SEQ_LEN);
  Serial.printf("pack : %.2f MB mapped | core Q%d + table Q%d g%d\n",
                g_pack_bytes / 1048576.0, M.h.core_bits, M.h.table_bits, M.h.group);
  Serial.printf("build: bytes=%u fp=%08x | ~%.2fM params, all dense-multiplied "
                "per token\n", (unsigned)g_pack_bytes, fp, g_stored_params / 1e6);

  if (!state_alloc_device()) { Serial.println("FATAL: state alloc"); return; }
  stage_to_psram();

  // BOS/EOS kept-row ids (original ids 1 and 2)
  for (int i = 0; i < M.h.vocab; i++) {
    if (M.remap[i] == 1) g_bos_row = i;
    if (M.remap[i] == 2) g_eos_row = i;
  }
  if (g_bos_row < 0) { Serial.println("FATAL: BOS not in kept set"); return; }
  if (M.h.vocab != VOCAB_N) {
    Serial.printf("FATAL: vocab.h has %d rows, pack has %d: reflash via "
                  "flash_record.sh so they regenerate together\n",
                  VOCAB_N, M.h.vocab);
    return;
  }

  Serial.printf("sram free %u KB | psram free %.2f MB\n",
                (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
                heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1048576.0);

  rng = esp_random() | 1ull;

  // Worker on core 0 (gen task runs on core 1). Hooked only once it exists.
  main_h = NULL;   // set by gen_task before its first forward
  if (xTaskCreatePinnedToCore(worker_main, "mv", 8 * 1024, NULL, 2,
                              &worker_h, 0) == pdPASS) {
    rq_matvec_hook = mv_par;
  } else {
    Serial.println("dual-core worker failed; running single core");
  }
#if USE_DISPLAY
  int disp_now = oled_ok ? 1 : 0;
#else
  int disp_now = 0;
#endif
  Serial.printf("config: dual_core=%d display=%d kv16=%d seq=%d\n",
                worker_h != NULL, disp_now, RQ_KV16, SEQ_LEN);

  xTaskCreatePinnedToCore(gen_task, "gen", GEN_TASK_STACK, NULL, 1, NULL, 1);
}

void loop() { vTaskDelay(pdMS_TO_TICKS(10000)); }
