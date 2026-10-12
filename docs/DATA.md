# Data Sources

## PhyWorld Collision Evaluation Set

**Purpose:** External validation data for collision-state parsing, metric compatibility, and later transfer experiments.

**Paper:** [How Far Is Video Generation from World Model: A Physical Law Perspective](https://arxiv.org/abs/2411.02385)

**Code:** [phyworld/phyworld](https://github.com/phyworld/phyworld)

**Dataset:** [magicr/phyworld](https://huggingface.co/datasets/magicr/phyworld)

**License:** Apache-2.0, as stated by the source repository. Dataset-specific terms should be rechecked before redistribution.

### Download

The dataset bytes are not committed to this repository.

```bash
mkdir -p data/external/phyworld
curl -L --fail \
  'https://huggingface.co/datasets/magicr/phyworld/resolve/main/id_ood_data/collision_eval.hdf5?download=true' \
  -o data/external/phyworld/collision_eval.hdf5
```

### Integrity

```text
File: id_ood_data/collision_eval.hdf5
Size: 15,832,960 bytes
SHA-256: 63904280119e7f73af782713c64350ebe9c4bf318ca93c1a0130e5c0714ad865
```

Validate the file and write a machine-readable manifest:

```bash
uv run python scripts/inspect_phyworld.py
```

### Observed Schema

| Group | Structure |
|---|---|
| `init_streams` | Three subgroups containing initial radius and velocity parameters |
| `position_streams` | Ground-truth arrays shaped `[examples, 32, 2, 2]` |
| `video_streams` | Variable-length compressed video byte streams |

Initial inspection found:

- 1,635 examples across three groups
- 32 state frames per example
- Radius range: 0.3–2.0 in source-world units
- Velocity range: 0.0–6.0 in source-world units
- Finite ground-truth positions for every inspected example

### Use in This Project

The initial use is structural and metric validation:

1. Verify that the loader preserves example alignment across video, state, and initialization groups.
2. Compare the repository’s state-error conventions with the source evaluation implementation.
3. Build an external collision transfer set after the controlled local pilot is stable.
4. Keep PhyWorld examples outside all controlled exposure-training branches unless a separate experiment explicitly registers their use.

### Citation

```bibtex
@article{kang2025phyworld,
  title={How Far Is Video Generation from World Model: A Physical Law Perspective},
  author={Kang, Bingyi and Yue, Yang and Lu, Rui and Lin, Zhijie and Zhao, Yang and Wang, Kaixin and Gao, Huang and Feng, Jiashi},
  journal={Proceedings of the 42nd International Conference on Machine Learning},
  year={2025}
}
```
