
### 1_data.md

Run:

```shell
`poetry run python sc_qrels/download_alice.py`
```

output:

```shell
data/raw/alice_in_wonderland.txt
```

### 2_prepare-alice-docs.md

The following script has been run once for this documents to find what should be normalized, it printed the output in the terminal. The output has been using to integrate in the following script the right normalization. So, this script is not part of the pipeline and, probably must be inlcuded in the `utils.py`.

```shell
sc_qrels/scan_unicode_characters.py
```

Run:

```shell
sc_qrels/prepare_alice_docs.py
```

output:

```shell
data/processed/documents
alice:ch01.json
alice:ch02.json
alice:ch03.json
alice:ch04.json
alice:ch05.json
alice:ch06.json
alice:ch07.json
alice:ch08.json
alice:ch09.json
alice:ch10.json
alice:ch11.json
alice:ch12.json
```

where each fine hsa the following format:

```shell
{
  "docid": "alice:ch01",
  "title": "CHAPTER I. Down the Rabbit-Hole",
  "text": "Alice was beginning to get very tired of sitting by her sister on the bank, ..."
}
```


