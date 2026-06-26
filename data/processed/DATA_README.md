# Processed Data README

Large processed replay tensors are not committed to git, but the expected directory structure is fixed.

## Expected layout

```text
data/processed/
├── battles/                # generated processed battle tensors
├── metadata.json           # dataset metadata
├── perspective_map.json    # perspective ID -> base battle ID mapping
├── priors.json             # processing-time metagame priors
└── vocabs/
    ├── abilities.json
    ├── actions.json
    ├── gen3ou/
    │   ├── abilities.json
    │   ├── actions.json
    │   ├── items.json
    │   ├── moves.json
    │   ├── species.json
    │   ├── status.json
    │   ├── terrain.json
    │   ├── types.json
    │   └── weather.json
    ├── items.json
    ├── moves.json
    ├── species.json
    ├── status.json
    ├── terrain.json
    ├── types.json
    └── weather.json
```

## How to populate it

1. Download raw replay files into `data/raw/`.
2. Run:

```bash
python scripts/process_dataset.py --input-dir data/raw --output-dir data/processed --generation gen3ou
```

## Notes

- The vocabularies in this repository are preserved because they are part of the training interface.
- The `battles/` directory is expected to be generated locally.
