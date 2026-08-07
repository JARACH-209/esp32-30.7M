#!/usr/bin/env bash
# Build and flash the stories42M record firmware + pack.
#
# Regenerates firmware/generated/vocab.h from keep_ids.txt on every run and
# cross-checks its row count against the pack header, so weights and decode
# table cannot drift. Compiles BEFORE flashing anything. Model pack goes to
# the `model` partition offset read from firmware/partitions.csv.
#
#   PORT=/dev/cu.usbmodemXXXX record42/flash_record.sh [pack.bin]
#
# Run from the repo root or from record42/: the script normalises.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PACK=${1:-record42.bin}
SKETCH=firmware
PARTS=$SKETCH/partitions.csv

[ -f "$PACK" ] || { echo "no pack at $HERE/$PACK: run quantize.py first" >&2; exit 1; }
[ -f keep_ids.txt ] || { echo "missing keep_ids.txt (rank_vocab.py)" >&2; exit 1; }
[ -f tokenizer.bin ] || { echo "missing tokenizer.bin" >&2; exit 1; }

# --- partition offset/size from the csv, never hardcoded ----------------------
read -r PART_OFFSET PART_SIZE <<EOF
$(awk -F',' '$1 ~ /^[[:space:]]*model[[:space:]]*$/ {
    gsub(/[[:space:]]/,"",$4); gsub(/[[:space:]]/,"",$5); print $4, $5 }' "$PARTS")
EOF
[ -n "${PART_OFFSET:-}" ] || { echo "no model row in $PARTS" >&2; exit 1; }

# --- validate the pack, read its vocab count ----------------------------------
VOCAB_IN_PACK=$(python3 - "$PACK" "$((PART_SIZE))" <<'PY'
import struct, sys, zlib
p, cap = sys.argv[1], int(sys.argv[2])
b = open(p, "rb").read()
magic, ver = struct.unpack_from("<II", b)
if magic != 0x32345152: sys.exit(f"{p}: not an RQ42 pack")
fields = struct.unpack_from("<IIiiiiiiiiiiiiIIIIIIIIII", b)
vocab = fields[7]
crc_stored = fields[-1]
norms_off = fields[15]
payload = b[norms_off:]
if zlib.crc32(payload) & 0xFFFFFFFF != crc_stored:
    sys.exit(f"{p}: crc mismatch: rebuild the pack")
if len(b) > cap:
    sys.exit(f"{p}: {len(b)} B exceeds the {cap} B model partition: "
             f"re-quantize with a smaller --keep")
print(vocab)
PY
) || { echo "$VOCAB_IN_PACK" >&2; exit 1; }

# --- regenerate firmware inputs ------------------------------------------------
cp runq.c runq.h "$SKETCH/"
python3 gen_vocab_header.py --tokenizer tokenizer.bin --keep keep_ids.txt \
    --out "$SKETCH/generated/vocab.h"
VOCAB_IN_HDR=$(grep -m1 "#define VOCAB_N" "$SKETCH/generated/vocab.h" | awk '{print $3}')
if [ "$VOCAB_IN_PACK" != "$VOCAB_IN_HDR" ]; then
  echo "vocab mismatch: pack $VOCAB_IN_PACK rows, header $VOCAB_IN_HDR." >&2
  echo "keep_ids.txt is not the list this pack was built from." >&2
  exit 1
fi

# --- tools ----------------------------------------------------------------------
command -v arduino-cli >/dev/null || { echo "missing arduino-cli" >&2; exit 1; }
find_esptool() {
  for c in esptool esptool.py; do command -v "$c" >/dev/null && { echo "$c"; return; }; done
  local data="${ARDUINO_DATA_DIR:-$HOME/Library/Arduino15}"
  [ -d "$data" ] || data="${ARDUINO_DATA_DIR:-$HOME/.arduino15}"
  ls -d "$data"/packages/esp32/tools/esptool_py/*/esptool 2>/dev/null | sort -V | tail -1
}
ESPTOOL=$(find_esptool)
[ -n "$ESPTOOL" ] || { echo "esptool not found" >&2; exit 1; }

PORT=${PORT:-$(ls /dev/cu.usbmodem* 2>/dev/null | head -1 || true)}
[ -n "$PORT" ] || { echo "no /dev/cu.usbmodem*; set PORT=..." >&2; exit 1; }

FQBN='esp32:esp32:esp32s3:UploadSpeed=921600,USBMode=hwcdc,CDCOnBoot=cdc,UploadMode=default,CPUFreq=240,FlashMode=qio,FlashSize=16M,PartitionScheme=custom,PSRAM=opi,DebugLevel=info'
BUILD_DIR="${TMPDIR:-/tmp}/record42-build"

echo "=== compile (Q4 runtime, KV16, -O3) ==="
arduino-cli compile --fqbn "$FQBN" \
  --build-property "compiler.optimization_flags=-O3" \
  --build-property "compiler.c.extra_flags=-DRQ_KV16=1" \
  --build-property "compiler.cpp.extra_flags=-DRQ_KV16=1" \
  --build-path "$BUILD_DIR" "$SKETCH" 2>&1 | tail -3

APP_BYTES=$(stat -f%z "$BUILD_DIR/$(basename $SKETCH).ino.bin" 2>/dev/null || \
            stat -c%s "$BUILD_DIR/$(basename $SKETCH).ino.bin")
APP_CAP=$((0x80000))
echo "app: $APP_BYTES B of $APP_CAP B partition"
[ "$APP_BYTES" -le "$APP_CAP" ] || {
  echo "app exceeds its 512KB partition: set USE_DISPLAY 0 in the sketch" >&2
  exit 1; }

FP=$(python3 - "$PACK" <<'PY'
import sys
d = open(sys.argv[1], "rb").read()
h = 2166136261
for b in d: h = ((h ^ b) * 16777619) & 0xFFFFFFFF
print("%08x" % h)
PY
)
BYTES=$(wc -c < "$PACK" | tr -d ' ')

echo
echo "=== about to flash ==="
printf "  pack   : %s (%s B, %s rows kept)\n" "$PACK" "$BYTES" "$VOCAB_IN_PACK"
printf "  port   : %s\n" "$PORT"
printf "  expect : fp=%s\n" "$FP"
echo

echo "=== flash pack -> $PORT @ $PART_OFFSET ==="
"$ESPTOOL" --chip esp32s3 --port "$PORT" --baud 921600 \
  write_flash "$PART_OFFSET" "$PACK" 2>&1 | tr '\r' '\n' | tail -2

echo "=== upload firmware ==="
arduino-cli upload -p "$PORT" --fqbn "$FQBN" --input-dir "$BUILD_DIR" "$SKETCH" 2>&1 | tail -1

echo
echo "flashed. expect at boot: 'build: bytes=$BYTES fp=$FP'"
echo "monitor:  arduino-cli monitor -p $PORT --config baudrate=115200"
