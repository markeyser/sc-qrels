# Step 1: Acquire and Prepare a Public-Domain Copy of *Alice’s Adventures in Wonderland*

This step involves downloading and preparing the source text for further processing. The version from [Project Gutenberg](https://www.gutenberg.org/) will be used due to its public-domain status and accessibility.

## 1.1 Download the Plain Text File

There are two options for acquiring the text:

### Option A: Manual Download

* URL: [https://www.gutenberg.org/cache/epub/11/pg11.txt](https://www.gutenberg.org/cache/epub/11/pg11.txt)
* Save the file locally as:

```
data/raw/alice_in_wonderland.txt
```

### Option B: Automated Download via Python

A script can be used to automate the download:

```python
# File: sc_qrels/download_alice.py

import os
import requests

os.makedirs("data/raw", exist_ok=True)
url = "https://www.gutenberg.org/cache/epub/11/pg11.txt"
response = requests.get(url)
with open("data/raw/alice_in_wonderland.txt", "w", encoding="utf-8") as f:
    f.write(response.text)
```

Execute the script with:

```bash
poetry run python sc_qrels/download_alice.py
```

---

## 1.2 Clean the Project Gutenberg Header and Footer

The downloaded file includes non-content sections such as licensing information and legal notices. These sections must be removed before further processing.

### Text Boundaries for Extraction:

* **Start** at the line:
  `"CHAPTER I. Down the Rabbit-Hole"`
* **End** at the line:
  `"THE END"`

A parser function will be implemented in the following step to isolate the relevant content.

---

## Summary of Actions

| Task                                               | Result                                           |
| -------------------------------------------------- | ------------------------------------------------ |
| Download *Alice in Wonderland* as a `.txt` file    | File saved at `data/raw/alice_in_wonderland.txt` |
| Verify presence of full text with chapter headings | \~27,000 words across 12 chapters                |
| Prepare for text cleaning and chapter parsing      | Proceed to the next step                         |

---


