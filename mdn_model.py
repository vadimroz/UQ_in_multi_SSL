import argparse
import itertools
import os
import sys
from math import ceil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import keras_mdn_layer as mdn
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from tqdm import tqdm

from Code.utilities import calc_azi_ele_err_degrees, print_results

DEFAULT_CONFIG_PATH = Path("configs/mdn/syn_srp_dnn_r400_snr15.yaml")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the MDN experiment from a YAML configuration.")
    parser.add_argument(
        "--config",
        type=Path,
        help=f"Path to the YAML configuration (default: {DEFAULT_CONFIG_PATH.name}).",
    )
    return parser.parse_args()


def parse_scalar(value: str):
    value = value.split(" #", 1)[0].rstrip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_simple_yaml(config_path: Path) -> dict:
    raw_lines = config_path.read_text().splitlines()
    lines = []
    for line in raw_lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        lines.append((len(line) - len(line.lstrip(" ")), line.strip()))

    root = {}
    stack = [(-1, root)]
    for index, (indent, stripped) in enumerate(lines):
        while indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]
        if stripped.startswith("- "):
            parent.append(parse_scalar(stripped[2:].strip()))
            continue

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            parent[key] = parse_scalar(value)
            continue

        next_is_list = False
        for next_indent, next_stripped in lines[index + 1:]:
            if next_indent > indent:
                next_is_list = next_stripped.startswith("- ")
                break
        container = [] if next_is_list else {}
        parent[key] = container
        stack.append((indent, container))

    return root


def load_config(config_path: Path) -> dict:
    try:
        import yaml
    except ModuleNotFoundError:
        return load_simple_yaml(config_path)

    with config_path.open("r") as stream:
        return yaml.safe_load(stream)


def resolve_config_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_template_path(path_template: str, values: dict) -> Path:
    try:
        formatted_path = path_template.format(**values)
    except KeyError as error:
        raise KeyError(f"Missing value required by path template: {error.args[0]}") from error
    return resolve_config_path(formatted_path)


def configure_runtime(config: dict) -> None:
    global DEFAULT_GPU_ID, MDN_DEVICE
    global DEFAULT_IN_CHANNELS, DEFAULT_NUM_SPEAKERS, DEFAULT_NUM_COMPONENTS, DEFAULT_OUTPUT_DIM
    global REVERB, SEED, iterations, alpha, REGION_SHAPE, verbose
    global AZ_RANGE, EL_RANGE, AZ_BINS, EL_BINS, AZ_BIN_SIZE, EL_BIN_SIZE

    dataset = config["dataset"]
    experiment = config["experiment"]
    model = config["model"]
    calibration = config["calibration"]
    grid = config["grid"]
    runtime = config["runtime"]

    DEFAULT_GPU_ID = int(runtime["gpu_id"])
    MDN_DEVICE = str(runtime["device"]).strip().lower()
    os.environ["OMP_NUM_THREADS"] = str(int(runtime["cpu_threads"]))

    DEFAULT_IN_CHANNELS = int(model["input_channels"])
    DEFAULT_NUM_SPEAKERS = int(dataset["speakers"])
    DEFAULT_NUM_COMPONENTS = int(model["num_components"])
    DEFAULT_OUTPUT_DIM = int(model["output_dim_per_speaker"])
    REVERB = int(dataset["reverb_ms"])
    SEED = int(experiment["seed"])
    iterations = int(experiment["num_iterations"])
    alpha = float(calibration["alpha"])
    REGION_SHAPE = str(calibration["region_shape"]).lower()
    verbose = bool(experiment["verbose"])

    AZ_RANGE = tuple(float(value) for value in grid["azimuth_range_deg"])
    EL_RANGE = tuple(float(value) for value in grid["elevation_range_deg"])
    AZ_BINS = int(grid["azimuth_bins"])
    EL_BINS = int(grid["elevation_bins"])
    AZ_BIN_SIZE = float(grid["azimuth_bin_size_deg"])
    EL_BIN_SIZE = float(grid["elevation_bin_size_deg"])


def configure_gpu(gpu_id: int) -> None:
    physical_gpus = tf.config.list_physical_devices("GPU")
    if not physical_gpus:
        return
    if gpu_id < 0 or gpu_id >= len(physical_gpus):
        raise ValueError(f"runtime.gpu_id={gpu_id} is invalid; found {len(physical_gpus)} GPU(s).")
    try:
        tf.config.set_visible_devices(physical_gpus[gpu_id], "GPU")
    except RuntimeError as error:
        raise RuntimeError("GPU visibility must be configured before TensorFlow initializes it.") from error


_DEFAULT_CONFIG = load_config(resolve_config_path(DEFAULT_CONFIG_PATH))
configure_runtime(_DEFAULT_CONFIG)


def extract_ordered_components(mu, sigma, pi_probs, n_samples):
    mu1_list, mu2_list = [], []
    sigma1_list, sigma2_list = [], []

    for s in range(n_samples):
        curr_mu = mu[s]
        curr_sigma = sigma[s]
        curr_pi = pi_probs[s]

        best_comp = np.argmax(curr_pi)

        spk_mu    = curr_mu   [best_comp*2 : best_comp*2 + 2, :]
        spk_sigma = curr_sigma[best_comp*2 : best_comp*2 + 2, :]

        order = np.argsort(spk_mu[:, 0])  # sort by azimuth
        mu1_list.append(spk_mu[order[0]])
        mu2_list.append(spk_mu[order[1]])
        sigma1_list.append(spk_sigma[order[0]])
        sigma2_list.append(spk_sigma[order[1]])

    return (np.array(mu1_list), np.array(mu2_list),
            np.array(sigma1_list), np.array(sigma2_list))


def find_joint_thresholds(scores: np.ndarray, target_coverage: float) -> tuple[float, float]:
    az_scores = scores[:, 0]
    el_scores = scores[:, 1]

    candidates = np.linspace(0.0, 1.0, 1001)
    az_thresh = el_thresh = None
    for t in candidates:
        az_t = np.quantile(az_scores, t)
        el_t = np.quantile(el_scores, t)
        coverage = np.mean((az_scores <= az_t) & (el_scores <= el_t))
        if coverage >= target_coverage:
            az_thresh, el_thresh = az_t, el_t
            break

    if az_thresh is None:
        az_thresh = np.quantile(az_scores, 1.0)
        el_thresh = np.quantile(el_scores, 1.0)

    return az_thresh, el_thresh


def compute_rect_area_bins(
    est_deg: np.ndarray,
    sig_deg: np.ndarray,
    az_thresh: float,
    el_thresh: float,
    az_range: tuple[float, float] | None = None,
    el_range: tuple[float, float] | None = None,
) -> float:
    az_range = AZ_RANGE if az_range is None else az_range
    el_range = EL_RANGE if el_range is None else el_range
    az_span = az_thresh * abs(sig_deg[0])
    el_span = el_thresh * abs(sig_deg[1])

    az_min = np.clip(est_deg[0] - az_span, az_range[0], az_range[1])
    az_max = np.clip(est_deg[0] + az_span, az_range[0], az_range[1])
    el_min = np.clip(est_deg[1] - el_span, el_range[0], el_range[1])
    el_max = np.clip(est_deg[1] + el_span, el_range[0], el_range[1])

    az_bins_cov = max(0.0, (az_max - az_min) / AZ_BIN_SIZE)
    el_bins_cov = max(0.0, (el_max - el_min) / EL_BIN_SIZE)

    return int(az_bins_cov * el_bins_cov)


def compute_ellipse_area_bins(
    est_deg: np.ndarray,
    sig_deg: np.ndarray,
    radius: float,
    az_range: tuple[float, float] | None = None,
    el_range: tuple[float, float] | None = None,
    az_bins: int | None = None,
    el_bins: int | None = None,
) -> float:
    az_range = AZ_RANGE if az_range is None else az_range
    el_range = EL_RANGE if el_range is None else el_range
    az_bins = AZ_BINS if az_bins is None else az_bins
    el_bins = EL_BINS if el_bins is None else el_bins
    az_radius = radius * abs(sig_deg[0])
    el_radius = radius * abs(sig_deg[1])
    if az_radius == 0 or el_radius == 0:
        return 0.0

    az_bin_size = AZ_BIN_SIZE
    el_bin_size = EL_BIN_SIZE

    az_centers = az_range[0] + az_bin_size * np.arange(az_bins)
    el_centers = el_range[0] + el_bin_size * np.arange(el_bins)

    az_grid, el_grid = np.meshgrid(az_centers, el_centers, indexing="xy")
    inside = ((az_grid - est_deg[0]) / az_radius) ** 2 + ((el_grid - est_deg[1]) / el_radius) ** 2 <= 1.0
    return float(np.count_nonzero(inside))


def plot_prediction_map(
    likelihood_map: np.ndarray,
    true_deg: list[np.ndarray],
    est_deg: list[np.ndarray],
    sig_deg: list[np.ndarray],
    region_shape: str,
    thresholds: list[tuple[float, float] | float],
    areas: list[float],
    az_range: tuple[float, float] | None = None,
    el_range: tuple[float, float] | None = None,
):
    az_range = AZ_RANGE if az_range is None else az_range
    el_range = EL_RANGE if el_range is None else el_range
    # print(f"True deg:{true_deg}")
    print(f"Est. deg:{est_deg}")

    az_centers = az_range[0] + AZ_BIN_SIZE * (0.5 + np.arange(AZ_BINS))
    el_centers = el_range[0] + EL_BIN_SIZE * (0.5 + np.arange(EL_BINS))

    def snap_to_centers(point_deg: np.ndarray) -> np.ndarray:
        az_idx = int(np.clip(np.round((point_deg[0] - az_range[0]) / AZ_BIN_SIZE - 0.5), 0, AZ_BINS - 1))
        el_idx = int(np.clip(np.round((point_deg[1] - el_range[0]) / EL_BIN_SIZE - 0.5), 0, EL_BINS - 1))
        return np.array([az_centers[az_idx], el_centers[el_idx]])

    likelihood_map = np.flipud(likelihood_map)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(
        likelihood_map,
        extent=[az_range[0], az_range[1], el_range[0], el_range[1]],
        interpolation="bilinear",
    )
    ax.set_xlim(az_range)
    ax.set_ylim(el_range)
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Elevation (deg)")
    ax.set_title("Likelihood map with true/estimated positions")

    colors = ["tab:red", "tab:orange"]
    for idx in range(len(true_deg)):
        true_point = true_deg[idx]
        est_point = est_deg[idx]
        est_sigma = sig_deg[idx]

        true_pt = snap_to_centers(true_point)
        est_pt = snap_to_centers(est_point)
        ax.scatter(true_pt[0], true_pt[1], marker="x", color=colors[idx], label=f"True {idx+1}")
        ax.scatter(est_pt[0], est_pt[1], marker="o", facecolors="none", edgecolors=colors[idx], label=f"Est {idx+1}")

        if region_shape == "ellipse":
            radius = thresholds[idx]
            width = 2 * radius * abs(est_sigma[0])
            height = 2 * radius * abs(est_sigma[1])
            patch = Ellipse((est_pt[0], est_pt[1]), width, height, fill=False, color=colors[idx])
        else:
            az_thresh, el_thresh = thresholds[idx]
            az_span = az_thresh * abs(est_sigma[0])
            el_span = el_thresh * abs(est_sigma[1])
            patch = Rectangle(
                (est_pt[0] - az_span, est_pt[1] - el_span),
                2 * az_span,
                2 * el_span,
                fill=False,
                color=colors[idx],
            )
        ax.add_patch(patch)
        ax.annotate(
            f"Area {idx + 1}: {areas[idx]:.1f}",
            xy=(est_pt[0], est_pt[1]),
            xytext=(6, 6),
            textcoords="offset points",
            color=colors[idx],
            fontsize=9,
        )

    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def resolve_device(requested_device: str | None = None) -> str:
    if requested_device is not None:
        device = str(requested_device).strip().lower()
        if device in {"cpu", "/cpu:0"}:
            return "/CPU:0"
        if device in {"cuda", "gpu", "/gpu:0"}:
            if not tf.config.list_physical_devices("GPU"):
                raise RuntimeError("GPU requested but no CUDA device is available")
            return "/GPU:0"
        return str(requested_device)

    if MDN_DEVICE in {"cpu"}:
        return "/CPU:0"

    if MDN_DEVICE in {"cuda", "gpu"}:
        if not tf.config.list_physical_devices("GPU"):
            raise RuntimeError("MDN_DEVICE=cuda requested but CUDA is not available")
        return "/GPU:0"

    if tf.config.list_physical_devices("GPU"):
        return "/GPU:0"

    return "/CPU:0"


checkpoint_dir = PROJECT_ROOT / "Exp"
best_model_path = checkpoint_dir / (
    f"mdn_best.speakers_{DEFAULT_NUM_SPEAKERS}_reverb_{REVERB}.weights.h5"
)
last_model_path = checkpoint_dir / "mdn_last.weights.h5"


def resolve_dataset_path(dataset_path: Path | str | None = None) -> Path:
    if dataset_path is None:
        default_dataset_path = PROJECT_ROOT / "data" / "mdn" / f"speakers_{DEFAULT_NUM_SPEAKERS}.npz"
        dataset_path = os.environ.get("MDN_DATASET_PATH", str(default_dataset_path))
    path = Path(dataset_path).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def load_dataset(dataset_path: Path | str | None = None) -> dict:
    resolved_path = resolve_dataset_path(dataset_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Dataset not found: {resolved_path}")
    with np.load(resolved_path, allow_pickle=True) as data:
        ds =  {key: data[key] for key in data.files}
        ds['speaker_pos'] = ds['speaker_pos'][:, :, ::-1]
        return ds


def normalize_likelihood_maps(likelihood_maps) -> np.ndarray:
    if isinstance(likelihood_maps, np.ndarray) and likelihood_maps.dtype == object:
        likelihood_maps = np.stack(likelihood_maps, axis=0)
    if likelihood_maps.ndim == 2:
        likelihood_maps = likelihood_maps[None, None, ...]
    elif likelihood_maps.ndim == 3:
        likelihood_maps = likelihood_maps[None, ...]
    elif likelihood_maps.ndim != 4:
        raise ValueError(f"Unexpected likelihood map shape: {likelihood_maps.shape}")
    return likelihood_maps


def prepare_likelihood_maps(likelihood_maps: np.ndarray) -> np.ndarray:
    likelihood_maps = normalize_likelihood_maps(likelihood_maps)
    if DEFAULT_IN_CHANNELS == 1:
        # min_vals = likelihood_maps.min(axis=(2, 3), keepdims=True)
        # max_vals = likelihood_maps.max(axis=(2, 3), keepdims=True)
        # likelihood_maps = (likelihood_maps - min_vals) / (max_vals - min_vals)
        return likelihood_maps[:, 0, ...]
    return likelihood_maps


def normalize_speaker_positions(speaker_positions, num_speakers: int) -> np.ndarray:
    if isinstance(speaker_positions, np.ndarray) and speaker_positions.dtype == object:
        speaker_positions = np.stack(speaker_positions, axis=0)
    else:
        speaker_positions = np.asarray(speaker_positions)

    if speaker_positions.ndim != 3 or speaker_positions.shape[-1] != 2:
        raise ValueError(
            f"Expected speaker positions of shape (B, S, 2), got {speaker_positions.shape}"
        )
    if speaker_positions.shape[1] != num_speakers:
        raise ValueError(
            f"Expected {num_speakers} speakers, got {speaker_positions.shape[1]}"
        )
    return speaker_positions


def build_mdn_model(
    in_channels: int, output_dim: int, num_components: int
) -> tf.keras.Model:
    if in_channels == 1:
        inputs = tf.keras.Input(shape=(EL_BINS, AZ_BINS), name="likelihood_maps")
        x = tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, axis=-1))(inputs)
    else:
        inputs = tf.keras.Input(shape=(in_channels, EL_BINS, AZ_BINS), name="likelihood_maps")
        x = tf.keras.layers.Lambda(lambda t: tf.transpose(t, [0, 2, 3, 1]))(inputs)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(2)(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(2)(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)

    outputs = mdn.MDN(output_dim, num_components)(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="mdn_model")


def build_mdn_model_from_dataset(mdn_dataset: dict) -> tf.keras.Model:
    _ = normalize_likelihood_maps(mdn_dataset["all_likelihood_maps"])
    num_speakers = int(mdn_dataset["speakers"])
    output_dim = num_speakers * DEFAULT_OUTPUT_DIM
    return build_mdn_model(
        in_channels=DEFAULT_IN_CHANNELS,
        output_dim=output_dim,
        num_components=DEFAULT_NUM_COMPONENTS,
    )


def build_default_mdn_model() -> tf.keras.Model:
    output_dim = DEFAULT_NUM_SPEAKERS * DEFAULT_OUTPUT_DIM
    return build_mdn_model(
        in_channels=DEFAULT_IN_CHANNELS,
        output_dim=output_dim,
        num_components=DEFAULT_NUM_COMPONENTS,
    )


def make_permutation_invariant_mdn_loss(
    base_mdn_loss_fn,
    num_speakers: int,
    output_dim_per_speaker: int | None = None,
):
    """Wraps an MDN loss function to calculate Permutation Invariant Loss (PIL)

    across all speaker identity variants.
    """
    output_dim_per_speaker = (
        DEFAULT_OUTPUT_DIM
        if output_dim_per_speaker is None
        else output_dim_per_speaker
    )
    speaker_indices = list(range(num_speakers))
    perms = list(itertools.permutations(speaker_indices))

    def pi_mdn_loss(y_true, y_pred):
        y_true_grouped = tf.reshape(y_true, [-1, num_speakers, output_dim_per_speaker])
        batch_size = tf.shape(y_true)[0]

        losses = []
        for perm in perms:
            permuted = tf.gather(y_true_grouped, perm, axis=1)
            permuted_flat = tf.reshape(permuted, [batch_size, -1])
            # Use sample-wise loss, not batch-averaged
            loss_val = base_mdn_loss_fn(permuted_flat, y_pred)  # shape: (batch,)
            losses.append(loss_val)

        stacked = tf.stack(losses, axis=0)  # (n_perms, batch)
        min_losses = tf.reduce_min(stacked, axis=0)  # (batch,)
        return tf.reduce_mean(min_losses)

    return pi_mdn_loss


def train_mdn(
    mdn_dataset: dict,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 2e-3,
    weight_decay: float = 1e-4,
    val_split: float = 0.2,
    seed: int | None = None,
    device: str | None = None,
) -> tf.keras.Model:
    device = resolve_device(device)
    seed = SEED if seed is None else seed

    likelihood_maps = prepare_likelihood_maps(mdn_dataset["all_likelihood_maps"])
    num_speakers = int(mdn_dataset["speakers"])
    speaker_positions = normalize_speaker_positions(
        mdn_dataset["speaker_pos"], num_speakers
    )
    output_dim = num_speakers * DEFAULT_OUTPUT_DIM

    x_tensor = likelihood_maps.astype(np.float32)
    y_tensor = speaker_positions.reshape(speaker_positions.shape[0], -1).astype(
        np.float32
    )

    num_samples = x_tensor.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    val_size = int(num_samples * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    num_train_samples = len(train_indices)
    steps_per_epoch = ceil(num_train_samples / batch_size)

    train_ds = (
        tf.data.Dataset.from_tensor_slices(
            (x_tensor[train_indices], y_tensor[train_indices])
        )
        .shuffle(len(train_indices))
        .batch(batch_size, drop_remainder=False)
    )

    val_ds = None
    if val_size > 0:
        val_ds = tf.data.Dataset.from_tensor_slices(
            (x_tensor[val_indices], y_tensor[val_indices])
        ).batch(batch_size, drop_remainder=False)

    with tf.device(device):
        model = build_mdn_model_from_dataset(mdn_dataset)

    # Get standard base MDN loss function
    base_loss_fn = mdn.get_mixture_loss_func(output_dim, DEFAULT_NUM_COMPONENTS)

    # Wrap loss function to handle Permutation Invariant calculations
    loss_fn = make_permutation_invariant_mdn_loss(
        base_loss_fn,
        num_speakers=num_speakers,
        output_dim_per_speaker=DEFAULT_OUTPUT_DIM,
    )

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1.0e-1,
        decay_steps=epochs * steps_per_epoch,
        alpha=1.0e-7,  # min LR
    )
    optimizer = tf.keras.optimizers.Adam(lr_schedule)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss_fn, optimizer=optimizer)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks = []
    if val_ds is not None:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                best_model_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            )
        )
    else:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                best_model_path,
                monitor="loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            )
        )
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            last_model_path,
            save_best_only=False,
            save_weights_only=True,
            verbose=0,
        )
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    return model


def load_mdn_model(
    model: tf.keras.Model | None = None, device: str | None = None
) -> tf.keras.Model:
    device = resolve_device(device)
    if model is None:
        model = build_default_mdn_model()
    with tf.device(device):
        if DEFAULT_IN_CHANNELS == 1:
            model.build(input_shape=(None, None, None))
        else:
            model.build(input_shape=(None, DEFAULT_IN_CHANNELS, None, None))
    model.load_weights(best_model_path)
    print(f"Loaded pretrained model from {best_model_path}")
    return model


if __name__ == "__main__":
    args = parse_arguments()
    selected_config = args.config or DEFAULT_CONFIG_PATH
    config_path = resolve_config_path(selected_config).resolve()
    config = load_config(config_path)

    dataset_config = config["dataset"]
    paths = config["paths"]
    experiment = config["experiment"]
    training_config = config["training"]
    runtime_config = config["runtime"]

    configure_runtime(config)
    calibration_fraction = float(experiment["calibration_fraction"])

    if DEFAULT_NUM_SPEAKERS != 2:
        raise ValueError("The current MDN evaluation pipeline supports exactly 2 speakers.")
    if not 0 < calibration_fraction < 1:
        raise ValueError("experiment.calibration_fraction must be between 0 and 1.")
    if not 0 < alpha < 1:
        raise ValueError("calibration.alpha must be between 0 and 1.")
    if REGION_SHAPE not in {"ellipse", "rect"}:
        raise ValueError("calibration.region_shape must be 'ellipse' or 'rect'.")

    configure_gpu(DEFAULT_GPU_ID)
    requested_device = str(runtime_config["device"]).lower()
    requested_device = None if requested_device == "auto" else requested_device

    calib_test_path = resolve_template_path(str(paths["evaluation_npz"]), dataset_config)
    training_path = resolve_template_path(str(paths["training_npz"]), dataset_config)
    best_model_path = resolve_template_path(str(paths["best_weights"]), dataset_config)
    last_model_path = resolve_template_path(str(paths["last_weights"]), dataset_config)
    checkpoint_dir = best_model_path.parent

    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    print(f"Using configuration: {config_path}")

    if not calib_test_path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {calib_test_path}")
    with np.load(calib_test_path, allow_pickle=True) as data:
        calib_test_ds = {key: data[key] for key in data.files}
    calib_test_ds['speaker_pos'] = calib_test_ds['speaker_pos'][:, :, ::-1]
    print("Loaded calib test dataset from:", calib_test_path)

    if os.path.exists(best_model_path):
        model = build_mdn_model_from_dataset(calib_test_ds)
        trained_model = load_mdn_model(model, device=requested_device)
    else:
        dataset = load_dataset(training_path)
        trained_model = train_mdn(
            dataset,
            epochs=int(training_config["epochs"]),
            batch_size=int(training_config["batch_size"]),
            learning_rate=float(training_config["learning_rate"]),
            weight_decay=float(training_config["weight_decay"]),
            val_split=float(training_config["validation_fraction"]),
            seed=SEED,
            device=requested_device,
        )

    mc_source_1, mc_source_2 = [], []
    mc_area_1, mc_area_2 = [], []
    mae_source_1, mae_source_2 = [], []

    for i_test in tqdm(range(iterations), desc="Iterations"):

        ds_size = calib_test_ds["all_likelihood_maps"].shape[0]
        indices = np.random.permutation(ds_size)
        cal_size = int(ds_size * calibration_fraction)
        cal_indices = indices[:cal_size]
        test_indices = indices[cal_size:]

        calib_likelihood_maps = prepare_likelihood_maps(
            calib_test_ds["all_likelihood_maps"][cal_indices]
        )
        calib_true_positions = calib_test_ds["speaker_pos"][cal_indices]
        calib_est_positions = calib_test_ds["all_estimated_positions"][cal_indices]

        calib_outputs = trained_model(calib_likelihood_maps, training=False)

        num_mixes = calib_true_positions.shape[1] * DEFAULT_OUTPUT_DIM
        output_dim = DEFAULT_NUM_COMPONENTS
        mu, sigma, pi_logits = mdn._split_mdn_params(calib_outputs, num_mixes, output_dim, )
        pi_probs = np.exp(pi_logits - np.max(pi_logits, axis=-1, keepdims=True))
        pi_probs = pi_probs / np.sum(pi_probs, axis=-1, keepdims=True)

        mu = mu.numpy()
        sigma = sigma.numpy()

        mu = np.reshape(mu, (-1, num_mixes, output_dim))
        sigma = np.reshape(sigma, (-1, num_mixes, output_dim))

        calib_mu1, calib_mu2, calib_sigma1, calib_sigma2 = extract_ordered_components(
            mu, sigma, pi_probs, len(cal_indices)
        )

        paired_true_deg, paired_est_deg, paired_sigma = [], [], []
        for i in range(len(cal_indices)):
            curr_true_pos = calib_true_positions[i]  # shape (2, 2)

            # Sort true positions by azimuth — same ordering as predictions
            order = np.argsort(curr_true_pos[:, 0])
            true_spk1 = curr_true_pos[order[0]]  # smaller azimuth
            true_spk2 = curr_true_pos[order[1]]  # larger azimuth

            paired_true_deg.append(np.rad2deg(true_spk1))
            paired_true_deg.append(np.rad2deg(true_spk2))

            paired_est_deg.append(np.rad2deg(calib_mu1[i]))
            paired_est_deg.append(np.rad2deg(calib_mu2[i]))

            paired_sigma.append(np.rad2deg(calib_sigma1[i]))
            paired_sigma.append(np.rad2deg(calib_sigma2[i]))

            if verbose:
                print(f"True {i} (deg): {np.rad2deg(true_spk1)} -> Matched est (deg): {np.rad2deg(calib_mu1[i])}")
                print(f"True {i} (deg): {np.rad2deg(true_spk2)} -> Matched est (deg): {np.rad2deg(calib_mu2[i])}")

                print("---------------------")

        paired_true_deg = np.array(paired_true_deg)  # shape (2*cal_size, 2) → (az, el)
        paired_est_deg = np.array(paired_est_deg)  # shape (2*cal_size, 2)
        paired_sigma = np.array(paired_sigma)

        # Per coordinate
        mae_az = np.mean(np.abs(paired_true_deg[:, 0] - paired_est_deg[:, 0]))
        mae_el = np.mean(np.abs(paired_true_deg[:, 1] - paired_est_deg[:, 1]))

        # Overall (across both az and el)
        mae_overall = np.mean(np.abs(paired_true_deg - paired_est_deg))

        if verbose:
            print(f"MAE Azimuth:   {mae_az:.2f} deg")
            print(f"MAE Elevation: {mae_el:.2f} deg")
            print(f"MAE Overall:   {mae_overall:.2f} deg")

        # Calibration
        # [calib samples, (azi score, ele scor)]
        source1_true, source2_true = paired_true_deg[::2], paired_true_deg[1::2]
        source1_est, source2_est = paired_est_deg[::2], paired_est_deg[1::2]
        source1_sig, source2_sig = paired_sigma[::2], paired_sigma[1::2]

        # Scores are in degrees
        score1 = np.abs(source1_true - source1_est) / source1_sig
        score2 = np.abs(source2_true - source2_est) / source2_sig

        # Quantiles (in degrees) per speaker based on region shape.
        target_cov = 1 - alpha
        if REGION_SHAPE == "ellipse":
            q1 = np.quantile(np.linalg.norm(score1, axis=1), target_cov)
            q2 = np.quantile(np.linalg.norm(score2, axis=1), target_cov)
        elif REGION_SHAPE == "rect":
            az_thresh1, el_thresh1 = find_joint_thresholds(score1, target_cov)
            az_thresh2, el_thresh2 = find_joint_thresholds(score2, target_cov)
        else:
            raise ValueError(f"Unsupported REGION_SHAPE: {REGION_SHAPE}")

        # Run Test
        test_likelihood_maps = prepare_likelihood_maps(calib_test_ds["all_likelihood_maps"][test_indices])
        test_true_positions = calib_test_ds["speaker_pos"][test_indices]
        test_est_positions = calib_test_ds["all_estimated_positions"][test_indices]

        outputs_test = trained_model(test_likelihood_maps, training=False)

        mu_test, sigma_test, pi_logits_test = mdn._split_mdn_params(
            outputs_test, num_mixes, output_dim
        )
        pi_probs_test = np.exp(
            pi_logits_test - np.max(pi_logits_test, axis=-1, keepdims=True)
        )
        pi_probs_test = pi_probs_test / np.sum(pi_probs_test, axis=-1, keepdims=True)

        mu_test = mu_test.numpy()
        sigma_test = sigma_test.numpy()
        mu_test = np.reshape(mu_test, (-1, num_mixes, output_dim))
        sigma_test = np.reshape(sigma_test, (-1, num_mixes, output_dim))

        # mu = [azimuth, elevation]
        test_mu1, test_mu2, test_sigma1, test_sigma2 = extract_ordered_components(
            mu_test, sigma_test, pi_probs_test, len(test_indices)
        )

        # For plotting purpose only
        if i_test <= 10:
            sample_idx = 0
            sample_map = test_likelihood_maps[sample_idx]
            sample_true = test_true_positions[sample_idx]
            order = np.argsort(sample_true[:, 0])
            true_spk1 = sample_true[order[0]]
            true_spk2 = sample_true[order[1]]

            est_deg = [
                np.rad2deg(test_mu1[sample_idx]),
                np.rad2deg(test_mu2[sample_idx]),
            ]
            sig_deg = [
                np.rad2deg(test_sigma1[sample_idx]),
                np.rad2deg(test_sigma2[sample_idx]),
            ]
            true_deg = [np.rad2deg(true_spk1), np.rad2deg(true_spk2)]

            if REGION_SHAPE == "ellipse":
                thresholds = [q1, q2]
                areas = [
                    compute_ellipse_area_bins(est_deg[0], sig_deg[0], q1),
                    compute_ellipse_area_bins(est_deg[1], sig_deg[1], q2),
                ]
            else:
                thresholds = [(az_thresh1, el_thresh1), (az_thresh2, el_thresh2)]
                areas = [
                    compute_rect_area_bins(est_deg[0], sig_deg[0], az_thresh1, el_thresh1),
                    compute_rect_area_bins(est_deg[1], sig_deg[1], az_thresh2, el_thresh2),
                ]

            # plot_prediction_map(
            #     sample_map,
            #     true_deg,
            #     est_deg,
            #     sig_deg,
            #     REGION_SHAPE,
            #     thresholds,
            #     areas,
            # )

        coverage_cnt_spk1, coverage_cnt_spk2 = 0, 0
        area_spk1, area_spk2 = [], []
        for i in range(len(test_indices)):
            curr_true_pos = test_true_positions[i]

            order = np.argsort(curr_true_pos[:, 0])
            true_spk1 = curr_true_pos[order[0]]
            true_spk2 = curr_true_pos[order[1]]

            sig1 = np.rad2deg(test_sigma1[i])
            sig2 = np.rad2deg(test_sigma2[i])

            err1 = np.rad2deg(true_spk1) - np.rad2deg(test_mu1[i])
            err2 = np.rad2deg(true_spk2) - np.rad2deg(test_mu2[i])

            # Here we have [azi, ele] pairs rather [ele, azi]
            mae_source_1.append(calc_azi_ele_err_degrees(true_spk1[::-1], test_mu1[i][::-1]))
            mae_source_2.append(calc_azi_ele_err_degrees(true_spk2[::-1], test_mu2[i][::-1]))

            score1_test = np.abs(err1) / sig1
            score2_test = np.abs(err2) / sig2

            if REGION_SHAPE == "ellipse":
                if np.linalg.norm(score1_test) <= q1:
                    coverage_cnt_spk1 += 1
                if np.linalg.norm(score2_test) <= q2:
                    coverage_cnt_spk2 += 1

                area_spk1.append(compute_ellipse_area_bins(np.rad2deg(test_mu1[i]), sig1, q1))
                area_spk2.append(compute_ellipse_area_bins(np.rad2deg(test_mu2[i]), sig2, q2))
            else:
                if (score1_test[0] <= az_thresh1) and (score1_test[1] <= el_thresh1):
                    coverage_cnt_spk1 += 1
                if (score2_test[0] <= az_thresh2) and (score2_test[1] <= el_thresh2):
                    coverage_cnt_spk2 += 1

                area_spk1.append(
                    compute_rect_area_bins(
                        np.rad2deg(test_mu1[i]),
                        sig1,
                        az_thresh1,
                        el_thresh1,
                    )
                )
                area_spk2.append(
                    compute_rect_area_bins(
                        np.rad2deg(test_mu2[i]),
                        sig2,
                        az_thresh2,
                        el_thresh2,
                    )
                )

        avg_area_spk1 = float(np.mean(area_spk1))
        avg_area_spk2 = float(np.mean(area_spk2))

        if verbose:
            print(f"Iter {i_test}: Coverage spk1: {coverage_cnt_spk1 / len(test_indices):.3f}")
            print(f"Iter {i_test}: Coverage spk2: {coverage_cnt_spk2 / len(test_indices):.3f}")
            print(f"Iter {i_test}: Avg area spk1 (bins^2): {avg_area_spk1:.2f}")
            print(f"Iter {i_test}: Avg area spk2 (bins^2): {avg_area_spk2:.2f}")
            print("------")

        mc_source_1.append(coverage_cnt_spk1 / len(test_indices))
        mc_source_2.append(coverage_cnt_spk2 / len(test_indices))
        mc_area_1.append(avg_area_spk1)
        mc_area_2.append(avg_area_spk2)

    mae_source_2 = np.mean(np.array(mae_source_2), axis=0)
    mae_source_1 = np.mean(np.array(mae_source_1), axis=0)

    grid_size = AZ_BINS * EL_BINS
    miscoverage = np.array([
        [1.0 - np.mean(mc_source_1)],
        [1.0 - np.mean(mc_source_2)],
    ])
    area_percent = np.array([
        [np.mean(mc_area_1) / grid_size * 100.0],
        [np.mean(mc_area_2) / grid_size * 100.0],
    ])
    mean_angular_error = np.array([
        [np.mean(mae_source_1)],
        [np.mean(mae_source_2)],
    ])

    print_results(
        miscoverage_array=miscoverage,
        area_array=area_percent,
        mean_angular_error=mean_angular_error,
        speakers=DEFAULT_NUM_SPEAKERS,
        significance_level=[alpha],
        area_unit="% of grid",
        title="MDN evaluation summary",
    )
