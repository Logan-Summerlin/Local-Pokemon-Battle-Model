"""Phase 4 data pipeline modules.

Modules:
- replay_parser: load Metamon-style replay files
- observation: build hidden-information-safe first-person observations
- tensorizer: convert observations into fixed tensors
- dataset: save/load processed tensorized battles
- auxiliary_labels: build auxiliary hidden-info targets
- base_stats: species base-stat lookup used by the observation pipeline
- priors: metagame prior aggregation used during processing
"""
