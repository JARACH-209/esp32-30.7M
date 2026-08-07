#!/usr/bin/env python3
"""Build the kept-vocabulary list from a generated corpus.

Reads token-id files produced by gen.c, counts frequencies, keeps the top K
(BOS/EOS/UNK and the 256 byte-fallback tokens are force-included by
quantize.py regardless, so any prompt still encodes). Prints coverage stats so
the trim's cost is a measured number, not a hope.

  ./gen stories42M.bin 40000 1.0 1 > ids_1.txt
  ./gen stories42M.bin 40000 1.0 2 > ids_2.txt
  ./gen stories42M.bin 40000 1.0 3 > ids_3.txt
  python3 rank_vocab.py ids_*.txt --keep 12000 -o keep_ids.txt
"""

import argparse
import struct
from collections import Counter
from pathlib import Path


def tokenizer_scores(path):
    """SentencePiece scores from llama2.c tokenizer.bin: the tokenizer's own
    trained frequency ranking. Used to fill the kept set past what a sampled
    corpus happens to touch: a TinyStories model SAYS ~3k distinct tokens but
    KNOWS far more, and the record claim counts stored rows the tied head
    multiplies every token, so filling with the most-frequent English tokens
    keeps every added row both honest and useful (better prompt coverage)."""
    raw = Path(path).read_bytes()
    off = 4                                        # skip max_token_length
    scores = []
    while off < len(raw):
        (score,) = struct.unpack_from("<f", raw, off)
        (ln,) = struct.unpack_from("<i", raw, off + 4)
        scores.append(score)
        off += 8 + ln
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="+", help="token-id files from gen.c")
    ap.add_argument("--keep", type=int, default=12000)
    ap.add_argument("--tokenizer", help="tokenizer.bin; enables score-fill up "
                    "to --keep when the corpus alone has fewer distinct ids")
    ap.add_argument("-o", "--out", default="keep_ids.txt")
    args = ap.parse_args()

    counts = Counter()
    total = 0
    for f in args.corpus:
        for line in Path(f).read_text().split():
            counts[int(line)] += 1
            total += 1

    ranked = counts.most_common()
    kept = [tid for tid, _ in ranked[: args.keep]]
    kept_set = set(kept)
    covered = sum(n for tid, n in ranked if tid in kept_set)
    filled = 0

    if args.tokenizer and len(kept) < args.keep:
        scores = tokenizer_scores(args.tokenizer)
        rest = sorted((i for i in range(len(scores)) if i not in kept_set),
                      key=lambda i: -scores[i])
        need = args.keep - len(kept)
        fill = rest[:need]
        kept.extend(fill)
        kept_set.update(fill)
        filled = len(fill)

    Path(args.out).write_text("\n".join(str(t) for t in sorted(kept)) + "\n")

    print(f"corpus         {total:,} tokens, {len(counts):,} distinct ids")
    print(f"kept           {len(kept):,} ids -> {args.out} "
          f"({len(kept) - filled:,} from corpus + {filled:,} score-filled)")
    print(f"coverage       {100.0 * covered / total:.3f}% of corpus tokens")
    if filled == 0 and len(counts) <= args.keep:
        print("note: corpus had fewer distinct ids than --keep and no "
              "--tokenizer was given to fill the difference")


if __name__ == "__main__":
    main()
