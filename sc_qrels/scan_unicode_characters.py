import os
import unicodedata
from collections import defaultdict

INPUT_FILE = "data/raw/alice_in_wonderland.txt"
MAX_EXAMPLES = 3

def is_control_or_ascii(c):
    return c == '\n' or c == '\t' or (32 <= ord(c) < 127)

def scan_unicode(text):
    seen = defaultdict(list)
    for i, char in enumerate(text):
        if is_control_or_ascii(char):
            continue
        cp = ord(char)
        name = unicodedata.name(char, "<unknown>")
        context = text[max(0, i-20):i+20].replace("\n", "\\n").replace("\t", "\\t")
        seen[(cp, char, name)].append(context)
    return seen

def report(seen):
    print(f"\nFound {len(seen)} non-ASCII characters:\n")
    for (cp, char, name), contexts in sorted(seen.items()):
        print(f"U+{cp:04X} '{char}' — {name}")
        for ctx in contexts[:MAX_EXAMPLES]:
            print(f"  ...{ctx}...")
        if len(contexts) > MAX_EXAMPLES:
            print(f"  (+ {len(contexts) - MAX_EXAMPLES} more)")
        print()

if __name__ == "__main__":
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    text = unicodedata.normalize("NFC", text)
    seen = scan_unicode(text)
    report(seen)
