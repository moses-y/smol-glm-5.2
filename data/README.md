# Prepared data

The trainer defaults to a project-local TinyStories dataset:

```text
data/tinystories/
├── metadata.json
├── tokenizer.json
├── train.bin
└── validation.bin
```

Create it from the repository root:

```bash
python3 scripts/prepare_tinystories.py \
  --output-dir data/tinystories
```

The binary shards, tokenizer, and metadata are ignored by Git. This README
remains tracked so the expected local layout is visible in a fresh clone.
Pass `--data-dir` to train from another prepared copy.
