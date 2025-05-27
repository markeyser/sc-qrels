# File: sc_qrels/download_alice.py

import os
import requests

os.makedirs("data/raw", exist_ok=True)
url = "https://www.gutenberg.org/cache/epub/11/pg11.txt"
response = requests.get(url)
with open("data/raw/alice_in_wonderland.txt", "w", encoding="utf-8") as f:
    f.write(response.text)