# SRP-DNN model

`SRP_DNN_best_model.tar` is the pretrained SRP-DNN checkpoint used by the
SRP-DNN localization pipeline. SRP-DNN uses a causal convolutional recurrent
network to estimate direct-path phase-difference sequences from microphone
pairs. Those estimates are combined using the steered-response-power
formulation to obtain a spatial spectrum for multiple-source localization.

The checkpoint uses PyTorch serialization; the pinned `torch` dependency is
listed in the project-level `requirements.txt`.

The method and original implementation come from:

- B. Yang, H. Liu, and X. Li, “SRP-DNN: Learning Direct-Path Phase Difference
  for Multiple Moving Sound Source Localization,” *ICASSP 2022*, Singapore,
  2022, pp. 721–725.
  [doi:10.1109/ICASSP43922.2022.9746624](https://doi.org/10.1109/ICASSP43922.2022.9746624)
- Original implementation: [BingYang-20/SRP-DNN](https://github.com/BingYang-20/SRP-DNN)

Please consult the upstream repository for the original implementation and
its licensing terms.
