# Adapted mixture-density network

The `mdn_*.weights.h5` files are checkpoints for this repository's adapted
mixture-density network, implemented in the project-level `mdn_model.py`. The
conceptual starting point is the Gaussian-mixture/MDN formulation proposed for
multi-source direction-of-arrival estimation with conformal coverage
guarantees by Khurjekar and Gerstoft.

The implementation in this repository is an adaptation, not an exact copy of
the authors' code. In particular, it operates on two-dimensional localization
likelihood maps, predicts azimuth/elevation mixture parameters, uses a
permutation-invariant training objective for multiple speakers, and calibrates
rectangular or elliptical prediction regions.

The model files are:

- `mdn_best.speakers_2_reverb_400.weights.h5`: selected checkpoint for the
  400 ms reverberation configuration.
- `mdn_best.speakers_2_reverb_700.weights.h5`: selected checkpoint for the
  700 ms reverberation configuration.
- `mdn_last.weights.h5`: final checkpoint written during MDN training.

The motivating paper is:

- I. D. Khurjekar and P. Gerstoft, “Multi-Source DOA Estimation With
  Statistical Coverage Guarantees,” *ICASSP 2024*, Seoul, Republic of Korea,
  2024, pp. 5310–5314.
  [doi:10.1109/ICASSP48485.2024.10446097](https://doi.org/10.1109/ICASSP48485.2024.10446097)

These MDN checkpoints belong to this repository's adaptation and should not
be presented as checkpoints released by the authors of the ICASSP 2024 paper.
