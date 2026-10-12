# Benchmark inputs

The `RCMP/` and `JSON/` directories contain 28 native zero-release instances from RCMPSPLIB, MPLIB1 and MPLIB2. The four development instances are `mpj60a2`, `mpj60a4`, `MPLIB1_Set1_1` and `MPLIB2_Set1_1`. The remaining 24 instances form the comparison set, as defined in `src/experiment_config.py`.

Conversion preserves project release dates, activity durations, precedence relations, renewable-resource demands and capacities. Dummy boundary nodes remain available to the decoding implementation.

To repeat conversion, run this command inside `data/dataset/`:

```sh
python convert_all.py
```

This recreates the converted JSON files. Public data access points and attribution are provided in the repository `REFERENCES.md`.
