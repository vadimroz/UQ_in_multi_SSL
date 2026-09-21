import argparse
from pathlib import Path

import numpy as np

from Code.crc_ssl import CoverageSet
from Code.plots import plot_roi_neighbours
from Code.utilities import create_save_directory, generate_random_splits, print_results


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = Path("configs/crc_ssl_n/syn_srp_phat_r400_snr15.yaml")

# filename = f'./data/{model_type}/Synthetic/Reverb_{reverb}_ms_SNR_{snr}_dB/speakers_{speakers}.npz'

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run CRC_SSL_N from a YAML configuration.")
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


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_dataset_path(path_template: str, dataset: dict) -> Path:
    try:
        path_value = path_template.format(
            speakers=int(dataset["speakers"]),
            snr_db=int(dataset["snr_db"]),
            reverb_ms=int(dataset["reverb_ms"]),
        )
    except KeyError as error:
        raise KeyError(f"Missing dataset value required by path template: {error.args[0]}") from error
    return resolve_project_path(path_value)


def main():
    args = parse_arguments()
    selected_config = args.config or DEFAULT_CONFIG_PATH
    config_path = resolve_project_path(selected_config).resolve()
    config = load_config(config_path)

    dataset = config["dataset"]
    paths = config["paths"]
    simulation = config["simulation"]
    calibration = config["calibration"]
    evaluation = config.get("evaluation", {})

    model_type = str(dataset["localization_method"])
    speakers = int(dataset["speakers"])
    num_iterations = int(simulation["num_iterations"])
    calibration_fraction = float(simulation["calibration_fraction"])
    seed = int(simulation["seed"])
    if not 0 < calibration_fraction < 1:
        raise ValueError("simulation.calibration_fraction must be between 0 and 1.")

    significance_level = np.asarray(calibration["significance_levels"], dtype=float)
    lambda_steps = int(calibration["lambda_steps"])
    lambda_list_ext = np.linspace(
        float(calibration.get("lambda_min", 0.0)),
        float(calibration.get("lambda_max", 1.0)),
        lambda_steps,
    )

    filename = resolve_dataset_path(str(paths["speakers_npz"]), dataset)
    if not filename.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {filename}")

    np.random.seed(seed)
    print(f"Using configuration: {config_path}")
    print(f"Loading dataset: {filename}")
    print(
        f"Current setup: {model_type}, {dataset.get('name', config_path.stem)}, {speakers} speakers, "
        f"{num_iterations} iterations"
    )

    data = np.load(filename, allow_pickle=True)
    speaker_pos = data["speaker_pos"]
    all_estimated_positions = data["all_estimated_positions"]
    all_likelihood_maps = data["all_likelihood_maps"]

    room_obj = data["rir_obj"].item()
    room = type("Room", (object,), room_obj)()
    grid_size = room.xl.size
    total_dataset_size = speaker_pos.shape[0]
    calibration_size = int(total_dataset_size * calibration_fraction)

    splits = generate_random_splits(
        total_samples=total_dataset_size,
        num_iterations=num_iterations,
        calib_size=calibration_size,
        num_lists=1,
        random_seed=seed,
    )
    folds_across_lists = list(zip(*splits))

    coverage_array = []
    area_array = []
    mean_angular_error_array = []
    plot = bool(evaluation.get("plot", False))
    plot_coverage_set = bool(evaluation.get("plot_coverage_set", False))
    test_plot = bool(evaluation.get("test_plot", False))
    output_dir = resolve_project_path(evaluation.get("output_dir", "outputs/crc_ssl_n"))

    for iteration_index in range(num_iterations):
        print(f"Fold {iteration_index + 1}/{num_iterations}")
        calib_index, test_index = folds_across_lists[iteration_index][0]

        if plot:
            dest_path = create_save_directory(
                str(output_dir / config_path.stem / f"fold_{iteration_index + 1}")
            )
        else:
            dest_path = None

        cov_set_obj = CoverageSet(
            true_position=speaker_pos[calib_index, ...],
            estimated_positions=all_estimated_positions[calib_index, :speakers, ...],
            likelihood_maps=all_likelihood_maps[calib_index, :speakers, ...],
            lambda_list=lambda_list_ext,
            room=room,
            path_=dest_path,
            plot_function=plot_roi_neighbours,
        )
        cov_set_obj.calibrate(plot=plot, plot_coverage_set=plot_coverage_set)

        coverage, area, metrics = cov_set_obj.test(
            test_sets=test_index.size,
            true_positions=speaker_pos[test_index, ...],
            estimated_positions=all_estimated_positions[test_index, :speakers, ...],
            likelihood_maps=all_likelihood_maps[test_index, :speakers, ...],
            significance_level=significance_level,
            test_plot=test_plot,
            collect_MAE_per_axis=True,
        )

        coverage_array.append(coverage)
        area_array.append(area / grid_size * 100)
        fold_angular_error = (
            np.asarray(metrics["mae_azimuth"], dtype=float)
            + np.asarray(metrics["mae_elevation"], dtype=float)
        ) / 2.0
        if fold_angular_error.shape != coverage.shape:
            raise ValueError(
                "Mean angular error shape does not match coverage shape: "
                f"{fold_angular_error.shape} != {coverage.shape}."
            )
        mean_angular_error_array.append(fold_angular_error)

    coverage_array = np.mean(coverage_array, axis=0)
    area_array = np.mean(area_array, axis=0)
    mean_angular_error_array = np.mean(mean_angular_error_array, axis=0)

    print_results(
        miscoverage_array=1 - coverage_array,
        area_array=area_array,
        significance_level=significance_level,
        mean_angular_error=mean_angular_error_array,
        area_unit="% of grid",
    )


if __name__ == "__main__":
    main()
