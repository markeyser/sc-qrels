# sc_qrels/prepare_alice_docs.py

import os
import re
import json
import unicodedata

RAW_PATH = "data/raw/alice_in_wonderland.txt"
OUT_DIR = "data/processed/documents"

def normalize(text: str) -> str:
    """
    Normalize Unicode characters and replace typographic artifacts with ASCII-safe alternatives.
    """
    text = unicodedata.normalize("NFC", text)

    replacements = {
        "\u201c": '"', "\u201d": '"',  # curly double quotes
        "\u2018": "'", "\u2019": "'",  # curly single quotes
        "\u2014": " -- ",              # em dash
        "\u2013": "-",                 # en dash
        "\u2026": "...",              # ellipsis
        "\u00a0": " ",                 # non-breaking space
        "\u2022": "*",                # bullet
        "\u2122": "",                 # trademark
        "\ufeff": "",                 # zero-width no-break space
    }

    for orig, repl in replacements.items():
        text = text.replace(orig, repl)

    # Collapse extra tabs/spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    return text

def load_raw_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def extract_main_text(raw_text):
    """
    Keep only from the first "CHAPTER I." up to (but not including) the final "THE END".
    """
    start = re.search(r"(?m)^CHAPTER\s+I\.", raw_text)
    end   = re.search(r"(?m)^THE END", raw_text)
    if not start or not end:
        raise RuntimeError("Unable to locate start/end markers in raw text.")
    return raw_text[start.start():end.start()]

def split_into_chapters(main_text):
    """
    Use a multiline regex to locate every chapter heading line,
    then slice out the text between successive headings.
    """
    chapter_pat = re.compile(
        r"(?m)^(CHAPTER\s+(?:I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII)\.\s*.*)$"
    )

    matches = list(chapter_pat.finditer(main_text))
    chapters = []

    for idx, match in enumerate(matches):
        title_line = match.group(1).strip()
        start_body = match.end()
        end_body = matches[idx+1].start() if idx+1 < len(matches) else len(main_text)
        body = main_text[start_body:end_body].strip()

        chapters.append({
            "docid": f"alice:ch{idx+1:02d}",
            "title": title_line,
            "text": body
        })

    return chapters

def save_chapters(chapters, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for ch in chapters:
        path = os.path.join(out_dir, f"{ch['docid']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ch, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    raw   = load_raw_text(RAW_PATH)
    raw   = normalize(raw)
    main  = extract_main_text(raw)
    docs  = split_into_chapters(main)
    save_chapters(docs, OUT_DIR)

    print(f"✔ Saved {len(docs)} chapters to '{OUT_DIR}'")
