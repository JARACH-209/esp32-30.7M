// Memory bandwidth benchmark for the 42M-parameter record attempt.
//
// The plan reads ~20MB of weights per generated token, split between
// memory-mapped flash and PSRAM. Token rate is therefore bandwidth, nothing
// else: tok/s = 1 / (flash_bytes/flash_MBps + psram_bytes/psram_MBps).
// Published numbers for this chip scatter from 10 to 40 MB/s for mmap'd
// flash, and the difference is "record lands at 1.4 tok/s" vs "sinks under
// 1". So: measure this exact board, then design to the measurement.
//
// What it measures, each pass:
//   flash mmap  - the model partition mapped into data address space, read
//                 sequentially two ways: a word-sum loop (what a naive kernel
//                 sees) and memcpy into SRAM (closer to peak; the SIMD kernels
//                 land between the two).
//   part read   - esp_partition_read into a 4KB SRAM buffer, the API voice.h
//                 used. Included to compare paths, not because the runtime
//                 would stream 13MB through it.
//   psram       - 4MB heap_caps_malloc'd buffer, same two read methods.
//   sram        - 64KB static buffer, as the ceiling reference.
//
// The flash region is read across a 13MB span so the 64KB data cache cannot
// fake the result: every pass is a cold streaming read, which is exactly what
// a >13MB weight scan does to the cache per token.
//
// FLASH IT (no wiring needed, replaces barista temporarily):
//   FQBN='esp32:esp32:esp32s3:USBMode=hwcdc,CDCOnBoot=cdc,PSRAM=opi,FlashSize=16M'
//   arduino-cli compile --fqbn "$FQBN" membench
//   arduino-cli upload -p /dev/cu.usbmodem1101 --fqbn "$FQBN" membench
//   arduino-cli monitor -p /dev/cu.usbmodem1101 --config baudrate=115200
//
// `scripts/deploy.sh barista` (or tinystories) restores the LLM afterwards.

#include <string.h>
#include "esp_partition.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "spi_flash_mmap.h"

#define FLASH_SPAN   (13u * 1024u * 1024u)   // what the 42M core scan covers
#define PSRAM_SPAN   (4u * 1024u * 1024u)
#define SRAM_SPAN    (64u * 1024u)
#define CHUNK        4096
#define REPEAT       3

static const uint8_t *flash_base = NULL;
static const esp_partition_t *part = NULL;
static uint8_t *psram_buf = NULL;
static uint8_t sram_buf[SRAM_SPAN];
static uint8_t chunk_buf[CHUNK];

// The word-sum loop. volatile accumulator so the compiler cannot delete the
// reads; sequential 32-bit words, which is what an unoptimised int4 unpack
// kernel does between SIMD loads.
static float sum_MBps(const uint8_t *base, size_t span) {
  volatile uint32_t acc = 0;
  int64_t t0 = esp_timer_get_time();
  for (int r = 0; r < REPEAT; r++) {
    const uint32_t *p = (const uint32_t *)base;
    for (size_t i = 0; i < span / 4; i++) acc += p[i];
  }
  int64_t us = esp_timer_get_time() - t0;
  (void)acc;
  return (float)((double)span * REPEAT / us);   // bytes/us == MB/s
}

// memcpy into SRAM in 4KB strides: the DMA-friendly upper bound for a
// streaming consumer.
static float memcpy_MBps(const uint8_t *base, size_t span) {
  int64_t t0 = esp_timer_get_time();
  for (int r = 0; r < REPEAT; r++)
    for (size_t off = 0; off + CHUNK <= span; off += CHUNK)
      memcpy(chunk_buf, base + off, CHUNK);
  int64_t us = esp_timer_get_time() - t0;
  return (float)((double)(span / CHUNK * CHUNK) * REPEAT / us);
}

static float partread_MBps(size_t span) {
  int64_t t0 = esp_timer_get_time();
  for (size_t off = 0; off + CHUNK <= span; off += CHUNK)
    if (esp_partition_read(part, off, chunk_buf, CHUNK) != ESP_OK) return -1;
  int64_t us = esp_timer_get_time() - t0;
  return (float)((double)(span / CHUNK * CHUNK) / us);
}

void setup() {
  Serial.setTxBufferSize(4096);
  Serial.begin(115200);
  delay(2500);
  Serial.println("\n=== MEMORY BANDWIDTH BENCHMARK ===");

  // Map RAW flash, no partition table involved. v1 of this benchmark mapped
  // "whatever data partition exists" and, uploaded with Arduino's default
  // table, that was nvs: a 20KB span that fit inside the 64KB data cache and
  // measured cache speed as flash speed. Mapping a fixed 13MB window starting
  // at 2MB makes the span physically incompressible by the cache, the same
  // region the model pack will occupy. Content is irrelevant to bandwidth.
  {
    const void *b;
    spi_flash_mmap_handle_t h;
    if (spi_flash_mmap(0x200000, FLASH_SPAN, SPI_FLASH_MMAP_DATA,
                       &b, &h) == ESP_OK) {
      flash_base = (const uint8_t *)b;
      Serial.printf("mapped raw flash: %u MB at offset 0x200000, addr %p\n",
                    FLASH_SPAN / 1048576u, flash_base);
    } else {
      Serial.println("raw flash mmap FAILED");
    }
  }
  part = esp_partition_find_first(ESP_PARTITION_TYPE_ANY,
                                  ESP_PARTITION_SUBTYPE_ANY, NULL);

  psram_buf = (uint8_t *)heap_caps_malloc(PSRAM_SPAN, MALLOC_CAP_SPIRAM);
  if (psram_buf) memset(psram_buf, 0xA5, PSRAM_SPAN);
  else Serial.println("PSRAM alloc FAILED");
  memset(sram_buf, 0x5A, sizeof(sram_buf));

  Serial.printf("cpu %d MHz | free sram %u KB | free psram %.2f MB\n\n",
                getCpuFrequencyMhz(),
                (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
                heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1048576.0);
}

void loop() {
  Serial.println("source        method     MB/s");
  Serial.println("------------  ---------  ------");
  if (flash_base) {
    Serial.printf("flash mmap    word-sum   %6.1f\n", sum_MBps(flash_base, FLASH_SPAN));
    Serial.printf("flash mmap    memcpy     %6.1f\n", memcpy_MBps(flash_base, FLASH_SPAN));
  }
  if (part) {
    size_t pspan = part->size < FLASH_SPAN ? part->size : FLASH_SPAN;
    Serial.printf("flash part    read4k     %6.1f  (span %u KB)\n",
                  partread_MBps(pspan), (unsigned)(pspan / 1024));
  }
  if (psram_buf) {
    Serial.printf("psram         word-sum   %6.1f\n", sum_MBps(psram_buf, PSRAM_SPAN));
    Serial.printf("psram         memcpy     %6.1f\n", memcpy_MBps(psram_buf, PSRAM_SPAN));
  }
  Serial.printf("sram          word-sum   %6.1f\n", sum_MBps(sram_buf, SRAM_SPAN));
  Serial.printf("sram          memcpy     %6.1f\n", memcpy_MBps(sram_buf, SRAM_SPAN));

  Serial.println("\nsend these numbers back; the flash mmap rows are the decision.");
  Serial.println("re-running in 10s...\n");
  delay(10000);
}
