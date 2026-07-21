#!/usr/bin/env python3
"""Extract the Google Benchmark JSON object from noisy stdin (e.g. d8 output).

d8 prints flag warnings / shell noise before the JSON. Scan every '{' and use
raw_decode until we find an object that has a "benchmarks" key, then emit it as
clean JSON on stdout. Exit non-zero if none is found.
"""
import json
import sys


def main() -> int:
    txt = sys.stdin.read()
    dec = json.JSONDecoder()
    i = 0
    n = len(txt)
    while i < n:
        j = txt.find("{", i)
        if j < 0:
            break
        try:
            obj, end = dec.raw_decode(txt, j)
        except json.JSONDecodeError:
            i = j + 1
            continue
        if isinstance(obj, dict) and "benchmarks" in obj:
            json.dump(obj, sys.stdout)
            return 0
        i = end
    return 1


if __name__ == "__main__":
    sys.exit(main())
