# Uncertainty Quantification and Risk Control for Multiple Sound Source Localization

This repository is the official implementation of the paper “Uncertainty
Quantification and Risk Control for Multiple Sound Source Localization.”

## Citation

If you use this implementation, please cite:

> V. Rozenfeld and B. Laufer-Goldshtein, "Uncertainty Quantification and Risk
> Control for Multiple Sound Source Localization," in IEEE Transactions on
> Audio, Speech and Language Processing, vol. 34, pp. 4231-4246, 2026,
> doi: 10.1109/TASLPRO.2026.3725584.

[Paper DOI](https://doi.org/10.1109/TASLPRO.2026.3725584)

## Included experiments

This branch contains only the code, configurations, datasets, and pretrained
checkpoint needed to run the default experiments in:

- `run_pt_ssl_u.py`
- `CRC_SSL_N.py`
- `mdn_model.py`

## Setup

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
python run_pt_ssl_u.py
python CRC_SSL_N.py
python mdn_model.py
```

Each script also accepts a configuration explicitly:

```bash
python run_pt_ssl_u.py --config configs/crc_ssl_u/locata_hybrid.yaml
python CRC_SSL_N.py --config configs/crc_ssl_n/syn_srp_phat_r400_snr15.yaml
python mdn_model.py --config configs/mdn/syn_srp_dnn_r400_snr15.yaml
```

All available configurations for these runners are included under `configs/`.
Every dataset and checkpoint referenced by those configurations is bundled
under the project-level `data/` and `Exp/` directories.

See [`Exp/README.md`](Exp/README.md) for model provenance, artifact details,
and citations for SRP-DNN and the adapted mixture-density network.
