import argparse
import glob
import itertools
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

from Code.PT_SSL_U.utilities import compute_Pareto_frontier, compute_p_values_wsr, compute_risks
from Code.utilities import generate_random_splits


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = Path("configs/crc_ssl_u/locata_hybrid.yaml")
GRID_SIZE = 37 * 73


def _natural_speakers_sort_key(filename):
    match = re.search(r"speakers_(\d+)\.npz$", os.path.basename(filename))
    if match is None:
        return (float("inf"), filename)
    return (int(match.group(1)), filename)


def calc_angular_errors_degrees(true_positions, estimated_positions):
    """Compute elevation and azimuth errors in degrees."""
    true_positions = np.rad2deg(np.asarray(true_positions))
    estimated_positions = np.rad2deg(np.asarray(estimated_positions))

    ele_true = true_positions[..., 0]
    azi_true = true_positions[..., 1]
    ele_est = estimated_positions[..., 0]
    azi_est = estimated_positions[..., 1]

    ele_err = np.mean(np.abs(ele_true - ele_est), axis=-1)
    azi_err = np.mean(np.abs((azi_true - azi_est + 180) % 360 - 180), axis=-1)
    return np.stack([azi_err, ele_err], axis=-1)


def match_estimated_positions(true_positions, estimated_positions):
    """Reorder estimated positions to minimize per-sample matching distance."""
    true_positions = np.asarray(true_positions)
    estimated_positions = np.asarray(estimated_positions)
    num_speakers = true_positions.shape[1]

    if num_speakers == 1:
        return estimated_positions

    permutations = list(itertools.permutations(range(num_speakers)))
    matched_estimated_positions = np.empty_like(estimated_positions)
    for sample_index in range(true_positions.shape[0]):
        best_permutation = min(
            permutations,
            key=lambda permutation: np.linalg.norm(
                true_positions[sample_index] - estimated_positions[sample_index, permutation, :],
                axis=1,
            ).sum(),
        )
        matched_estimated_positions[sample_index] = estimated_positions[sample_index, best_permutation, :]

    return matched_estimated_positions


def load_speakers_npz_files(pattern: str) -> pd.DataFrame:
    """Load and combine numeric speakers_<n>.npz files matching a glob."""
    numeric_speakers_file = re.compile(r"^speakers_(\d+)\.npz$")
    filenames = [
        filename
        for filename in glob.glob(pattern)
        if numeric_speakers_file.match(os.path.basename(filename))
    ]
    filenames.sort(key=_natural_speakers_sort_key)
    if not filenames:
        raise FileNotFoundError(f"No numeric speakers files matched pattern: {pattern}")

    frames = []
    for filename in filenames:
        curr_speakers = int(numeric_speakers_file.match(os.path.basename(filename)).group(1))
        with np.load(filename, allow_pickle=True) as data:
            true_positions = np.asarray(data["speaker_pos"])[:, :curr_speakers, :]
            estimated_positions = np.asarray(data["all_estimated_positions"])[:, :curr_speakers, :]
            estimated_positions = match_estimated_positions(true_positions, estimated_positions)
            angular_errors = calc_angular_errors_degrees(true_positions, estimated_positions).squeeze()
            sample_count = true_positions.shape[0]

            frames.append(
                pd.DataFrame(
                    {
                        "Sample": np.arange(sample_count),
                        "True_speakers": curr_speakers,
                        "azi_err": list(np.round(angular_errors[..., 0], 2)),
                        "ele_err": list(np.round(angular_errors[..., 1], 2)),
                    }
                )
            )

    return pd.concat(frames, ignore_index=True)


def merge_speaker_errors(loss_data: pd.DataFrame, speaker_errors: pd.DataFrame) -> pd.DataFrame:
    """Attach per-sample azimuth/elevation errors after Loss_Area."""
    speaker_error_columns = ["azi_err", "ele_err"]
    loss_data = loss_data.drop(columns=[col for col in speaker_error_columns if col in loss_data], errors="ignore")
    loss_data = loss_data.merge(
        speaker_errors,
        on=["Sample", "True_speakers"],
        how="left",
        validate="many_to_one",
    )

    if "Loss_Area" not in loss_data.columns:
        return loss_data

    remaining_columns = [column for column in loss_data.columns if column not in speaker_error_columns]
    loss_area_index = remaining_columns.index("Loss_Area")
    return loss_data[
        remaining_columns[: loss_area_index + 1]
        + speaker_error_columns
        + remaining_columns[loss_area_index + 1 :]
    ]


@dataclass
class IterationSplit:
    cal_opt: dict
    cal_test: dict
    test: dict


class DataManager:
    def __init__(self, loss_by_config: pd.DataFrame, kmax: int):
        self.loss_by_config = loss_by_config
        self.kmax = kmax

    def get_rows(self, sample_indices, speaker_id: int, config_index=None):
        mask = (
            self.loss_by_config["Sample"].isin(sample_indices)
            & (self.loss_by_config["True_speakers"] == speaker_id)
        )
        if config_index is not None:
            mask = mask & (self.loss_by_config["config_index"] == config_index)
        return self.loss_by_config[mask]

    def combine_rows(self, samples_by_speaker: dict, config_index=None):
        frames = [
            self.get_rows(sample_indices, speaker_id, config_index=config_index)
            for speaker_id, sample_indices in samples_by_speaker.items()
        ]
        return pd.concat(frames, ignore_index=True)


class OptimizationEngine:
    def __init__(
        self,
        data_manager: DataManager,
        kmax: int,
        alphas: np.ndarray,
        delta: float,
        lambda_grid: list,
        wsr_scale: float = 1.0,
    ):
        self.data_manager = data_manager
        self.kmax = kmax
        self.alphas = alphas
        self.delta = delta
        self.lambda_grid = lambda_grid
        self.wsr_scale = wsr_scale

    def run(self, split: IterationSplit) -> pd.DataFrame:
        cal_opt_combined = self.data_manager.combine_rows(split.cal_opt)
        opt_risks = compute_risks(cal_opt_combined, self.kmax)
        costs = opt_risks.loc[:, opt_risks.columns.str.startswith("Risk")].to_numpy()
        efficient_indices = compute_Pareto_frontier(costs)
        lambda_pareto = opt_risks[efficient_indices]

        pareto_combinations = opt_risks.loc[efficient_indices, "config_index"].to_numpy()
        pareto_set_losses = cal_opt_combined[cal_opt_combined["config_index"].isin(pareto_combinations)]
        p_values_opt_wsr = compute_p_values_wsr(
            pareto_set_losses,
            pareto_combinations,
            self.alphas,
            self.delta,
            self.kmax,
            scale=self.wsr_scale,
        )

        lambda_sorted = lambda_pareto.copy()
        lambda_sorted.loc[:, "p_values"] = p_values_opt_wsr
        lambda_sorted.sort_values(by="p_values", ascending=True, inplace=True)
        return lambda_sorted


class CalibrationEngine:
    def __init__(
        self,
        data_manager: DataManager,
        kmax: int,
        alphas: np.ndarray,
        delta: float,
        free_risks: list,
        wsr_scale: float = 1.0,
    ):
        self.data_manager = data_manager
        self.kmax = kmax
        self.alphas = alphas
        self.delta = delta
        self.free_risks = free_risks
        self.wsr_scale = wsr_scale

    def select_configuration(self, split: IterationSplit, optimization_result: pd.DataFrame) -> int:
        cal_test_combined = self.data_manager.combine_rows(split.cal_test)
        config_order_by_p_value = optimization_result["config_index"].tolist()
        cal_test_combined = cal_test_combined[
            cal_test_combined["config_index"].isin(config_order_by_p_value)
        ].copy()

        config_rank = {
            config_index: order_index
            for order_index, config_index in enumerate(config_order_by_p_value)
        }
        cal_test_combined["_config_rank"] = cal_test_combined["config_index"].map(config_rank)
        cal_test_combined.sort_values("_config_rank", kind="stable", inplace=True)
        cal_test_combined.drop(columns="_config_rank", inplace=True)

        cal_risks = compute_risks(cal_test_combined, self.kmax)
        cal_risks["_config_rank"] = cal_risks["config_index"].map(config_rank)
        cal_risks.sort_values("_config_rank", kind="stable", inplace=True)
        cal_risks.drop(columns="_config_rank", inplace=True)

        p_values_testing = compute_p_values_wsr(
            cal_test_combined,
            config_order_by_p_value,
            self.alphas,
            self.delta,
            self.kmax,
            scale=self.wsr_scale,
        )
        cal_risks.loc[:, "p_values"] = p_values_testing

        mask = cal_risks["p_values"].le(self.delta).cummin()
        lambda_rejected = cal_risks.loc[mask]
        costs_free = lambda_rejected.loc[:, lambda_rejected.columns.isin(self.free_risks)].to_numpy()
        efficient_indices = compute_Pareto_frontier(costs_free)
        lambda_star = lambda_rejected[efficient_indices]

        try:
            return lambda_star.loc[lambda_star["Risk_Area"].idxmin(), "config_index"]
        except Exception:
            print("****")
            return cal_risks.loc[cal_risks["p_values"].idxmin(), "config_index"]


class BaseTestingEngine:
    def __init__(self, data_manager: DataManager, kmax: int, calib_opt: int):
        self.data_manager = data_manager
        self.kmax = kmax
        self.calib_opt = calib_opt

    def _build_test_set(self, split: IterationSplit, chosen_config_index: int) -> pd.DataFrame:
        raise NotImplementedError

    def evaluate(self, split: IterationSplit, chosen_config_index: int, iteration_index: int) -> pd.Series:
        test_combined = self._build_test_set(split, chosen_config_index).copy()
        test_combined["Loss_Area"] = (
            test_combined["Loss_Area"] / test_combined["Estimated_speakers"].replace(0, np.nan)
        ).fillna(0) * (100.0 / GRID_SIZE)
        test_combined = test_combined.tail(297).reset_index(drop=True)

        result_columns = ["Loss_MC", "Loss_MD", "Loss_FA", "Loss_Area"]
        if {"azi_err", "ele_err"}.issubset(test_combined.columns):
            result_columns.extend(["azi_err", "ele_err"])
        result = test_combined[result_columns].mean()
        result["Iteration"] = iteration_index
        result["n_calibration"] = 2 * self.calib_opt
        return result


class LocataTestingEngine(BaseTestingEngine):
    def __init__(
        self,
        data_manager: DataManager,
        kmax: int,
        calib_opt: int,
        locata_test_loss: pd.DataFrame,
        samples_per_speaker: int = 10,
    ):
        super().__init__(data_manager, kmax, calib_opt)
        self.locata_test_loss = locata_test_loss
        self.samples_per_speaker = samples_per_speaker

    def _build_test_set(self, split: IterationSplit, chosen_config_index: int) -> pd.DataFrame:
        test_combined = self.locata_test_loss[
            self.locata_test_loss["config_index"] == chosen_config_index
        ]
        samples = []
        for speaker_id in range(self.kmax, 0, -1):
            speaker_rows = test_combined[test_combined["True_speakers"] == speaker_id]
            if len(speaker_rows) < self.samples_per_speaker:
                raise ValueError(
                    f"Configuration {chosen_config_index} has only {len(speaker_rows)} rows "
                    f"for {speaker_id} speaker(s); requested {self.samples_per_speaker}."
                )
            samples.append(speaker_rows.sample(n=self.samples_per_speaker, replace=False))
        return pd.concat(samples, axis=0)


class SyntheticTestingEngine(BaseTestingEngine):
    def _build_test_set(self, split: IterationSplit, chosen_config_index: int) -> pd.DataFrame:
        return self.data_manager.combine_rows(split.test, config_index=chosen_config_index)


class ExperimentRunner:
    def __init__(self, loss_by_config: pd.DataFrame, kmax: int, config: dict):
        self.loss_by_config = loss_by_config
        self.kmax = kmax
        self.config = config
        self.data_manager = DataManager(loss_by_config=loss_by_config, kmax=kmax)

    def _build_lambda_grid(self):
        mc_grids = [
            self.loss_by_config[f"Threshold_MC_{speaker_id}"].unique().tolist()
            for speaker_id in range(1, self.kmax + 1)
        ]
        md_grid = self.loss_by_config["Threshold_MD"].unique().tolist()
        mc_combinations = list(itertools.product(*mc_grids))
        return [(mc_vals, md_val) for mc_vals in mc_combinations for md_val in md_grid]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run PT_SSL_U experiment from YAML config.")
    parser.add_argument(
        "--config",
        type=Path,
        help=f"Path to the YAML configuration to run (default: {DEFAULT_CONFIG_PATH}).",
    )
    return parser.parse_args()


def resolve_config_path(config_path: Path | None) -> Path:
    resolved_path = (config_path or DEFAULT_CONFIG_PATH).expanduser()
    if not resolved_path.is_absolute():
        resolved_path = PROJECT_ROOT / resolved_path

    resolved_path = resolved_path.resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {resolved_path}")
    return resolved_path


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_scalar(value: str):
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


def resolve_simulation_sizes(config: dict) -> tuple[int, int, int, int]:
    simulation = config["simulation"]
    test_sets = int(simulation["test_sets"])

    if "calib_opt" in simulation and "calib_test" in simulation:
        calib_opt = int(simulation["calib_opt"])
        calib_test = int(simulation["calib_test"])
        total_samples = int(simulation.get("total_samples", calib_opt + calib_test + test_sets))
    elif simulation.get("calibration_split_policy") == "split_remaining_evenly":
        total_samples = int(simulation["total_samples"])
        calibration_total = total_samples - test_sets
        calib_opt = calibration_total // 2
        calib_test = calibration_total - calib_opt
    else:
        calib_opt = int(simulation["calib_opt"])
        calib_test = int(simulation["calib_test"])
        calibration_total = calib_opt + calib_test
        total_samples = calibration_total + test_sets

    return calib_opt, calib_test, test_sets, total_samples


def format_metric(value, digits: int = 4) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def format_iteration_result(
    iteration_index: int,
    num_iterations: int,
    result: pd.Series,
    alphas: np.ndarray,
) -> str:
    mc_status = "OK" if result["Loss_MC"] <= alphas[1] else "HIGH"
    md_status = "OK" if result["Loss_MD"] <= alphas[0] else "HIGH"
    parts = [
        f"[{iteration_index + 1:03d}/{num_iterations:03d}]",
        f"MC={format_metric(result['Loss_MC'])} ({mc_status})",
        f"MD={format_metric(result['Loss_MD'])} ({md_status})",
        f"FA={format_metric(result['Loss_FA'])}",
        f"Area={format_metric(result['Loss_Area'])}%",
    ]

    if {"azi_err", "ele_err"}.issubset(result.index):
        mean_angular_error = (result["azi_err"] + result["ele_err"]) / 2
        parts.append(f"mean_angular={format_metric(mean_angular_error, digits=2)} deg")

    return " | ".join(parts)


def print_summary_table(
    summary: dict,
    alphas: np.ndarray,
    delta: float,
    num_iterations: int,
) -> None:
    rows = [
        ("Mean false-alarm loss", format_metric(summary["mean_Loss_FA"])),
        ("Mean area (% of grid)", format_metric(summary["mean_Loss_Area"])),
        ("Mean miscoverage loss", format_metric(summary["MC_mean"])),
        ("Mean misdetection loss", format_metric(summary["MD_mean"])),
        (
            f"MC exceedances (> alpha_MC={alphas[1]:g})",
            f"{summary['count_Loss_MC_gt_alpha1']} / {num_iterations}",
        ),
        (
            f"MD exceedances (> alpha_MD={alphas[0]:g})",
            f"{summary['count_Loss_MD_gt_alpha0']} / {num_iterations}",
        ),
        ("Mean angular error (deg)", format_metric(summary["mean_angular_err"])),
    ]
    print("\nExperiment summary")
    print(tabulate(rows, headers=("Metric", "Value"), tablefmt="rounded_outline"))


class YamlExperimentRunner(ExperimentRunner):
    def __init__(self, loss_by_config: pd.DataFrame, kmax: int, config: dict, free_risks: list[str]):
        super().__init__(loss_by_config=loss_by_config, kmax=kmax, config=config)
        self.free_risks = free_risks

    def _build_iteration_split(self, iteration_folds) -> IterationSplit:
        cal_opt = {}
        cal_test = {}
        test = {}

        for speaker_id in range(1, self.kmax + 1):
            speaker_cal, speaker_test = iteration_folds[speaker_id - 1]
            speaker_cal_opt = np.sort(speaker_cal[: self.config["calib_opt"]])[: self.config["calib_opt"]]
            speaker_cal_test = np.sort(speaker_cal[self.config["calib_opt"] :])[: self.config["calib_test"]]

            cal_opt[speaker_id] = speaker_cal_opt
            cal_test[speaker_id] = speaker_cal_test
            test[speaker_id] = speaker_test

        return IterationSplit(cal_opt=cal_opt, cal_test=cal_test, test=test)

    def run(self):
        calibration_total = self.config["calib_opt"] + self.config["calib_test"]
        print(
            "Simulation split sizes: "
            f"calibration_total={calibration_total} "
            f"(cal_opt={self.config['calib_opt']}, cal_test={self.config['calib_test']}), "
            f"test={self.config['test_sets']}, total_samples={self.config['samples']}, "
            f"num_iterations={self.config['num_iterations']}"
        )

        lambda_grid = self._build_lambda_grid()
        splits = generate_random_splits(
            total_samples=self.config["samples"],
            num_iterations=self.config["num_iterations"],
            calib_size=self.config["samples"] - self.config["test_sets"],
            num_lists=self.kmax,
        )
        folds_across_lists = list(zip(*splits))

        optimization_engine = OptimizationEngine(
            data_manager=self.data_manager,
            kmax=self.kmax,
            alphas=self.config["alphas"],
            delta=self.config["delta"],
            lambda_grid=lambda_grid,
            wsr_scale=self.config["wsr_scale"],
        )
        calibration_engine = CalibrationEngine(
            data_manager=self.data_manager,
            kmax=self.kmax,
            alphas=self.config["alphas"],
            delta=self.config["delta"],
            free_risks=self.free_risks,
            wsr_scale=self.config["wsr_scale"],
        )
        if self.config["test_strategy"] == "locata_random_per_speaker":
            testing_engine = LocataTestingEngine(
                data_manager=self.data_manager,
                kmax=self.kmax,
                calib_opt=self.config["calib_opt"],
                locata_test_loss=self.config["locata_test_loss"],
                samples_per_speaker=self.config["locata_test_samples_per_speaker"],
            )
        else:
            testing_engine = SyntheticTestingEngine(
                data_manager=self.data_manager,
                kmax=self.kmax,
                calib_opt=self.config["calib_opt"],
            )

        test_results = []
        miscoverage_instances = 0
        midetect_instances = 0

        progress = tqdm(range(self.config["num_iterations"]), desc="Running iterations", unit="iter")
        for iteration_index in progress:
            split = self._build_iteration_split(folds_across_lists[iteration_index])
            optimization_result = optimization_engine.run(split)
            chosen_config_index = calibration_engine.select_configuration(split, optimization_result)
            res = testing_engine.evaluate(split, chosen_config_index, iteration_index)
            test_results.append(res)

            if res["Loss_MC"] > self.config["alphas"][1]:
                miscoverage_instances += 1
            if res["Loss_MD"] > self.config["alphas"][0]:
                midetect_instances += 1

            progress.set_postfix(
                {
                    "MC": format_metric(res["Loss_MC"]),
                    "MD": format_metric(res["Loss_MD"]),
                    "Area (%)": format_metric(res["Loss_Area"]),
                }
            )
            tqdm.write(
                format_iteration_result(
                    iteration_index=iteration_index,
                    num_iterations=self.config["num_iterations"],
                    result=res,
                    alphas=self.config["alphas"],
                )
            )

        return pd.DataFrame(test_results), miscoverage_instances, midetect_instances


def main():
    args = parse_arguments()
    config_path = resolve_config_path(args.config)
    print(f"Using configuration: {config_path}")
    config = load_config(config_path)

    paths = config["paths"]
    dataset = config["dataset"]
    calibration = config["calibration"]
    simulation = config["simulation"]
    evaluation = config.get("evaluation", {})

    np.random.seed(int(config.get("experiment", {}).get("seed", 1234567890)))

    calib_opt, calib_test, test_sets, total_samples = resolve_simulation_sizes(config)
    alphas = np.array([float(calibration["alpha_MD"]), float(calibration["alpha_MC"])])

    if alphas[0] != alphas[1]:
        raise ValueError("alpha_MC and alpha_MD must be same.")

    test_strategy = evaluation.get("test_strategy", "held_out")
    supported_test_strategies = {"held_out", "locata_random_per_speaker"}
    if test_strategy not in supported_test_strategies:
        raise ValueError(
            f"Unsupported evaluation.test_strategy {test_strategy!r}; "
            f"expected one of {sorted(supported_test_strategies)}."
        )

    loss_path = resolve_project_path(paths["loss_parquet"])
    speakers_glob = str(resolve_project_path(paths["speakers_glob"]))

    print(f"Loading dataset from {loss_path} ...")
    loss_by_config = pd.read_parquet(loss_path, engine="pyarrow")
    print(f"Loaded prefiltered loss_by_config with shape {loss_by_config.shape}.")

    print(f"Loading speakers data from {speakers_glob} ...")
    speakers_data = load_speakers_npz_files(speakers_glob)
    print(f"Loaded speakers data with shape {speakers_data.shape}.")
    loss_by_config = merge_speaker_errors(loss_by_config, speakers_data)
    print(f"Merged speakers data into loss_by_config; new shape {loss_by_config.shape}.")

    runner_config = {
        "test_on_locata": test_strategy == "locata_random_per_speaker",
        "use_locata": test_strategy == "locata_random_per_speaker",
        "num_iterations": int(simulation["num_iterations"]),
        "calib_opt": calib_opt,
        "calib_test": calib_test,
        "test_sets": test_sets,
        "samples": total_samples,
        "delta": float(calibration["delta"]),
        "wsr_scale": float(calibration.get("wsr_scale", 1.0)),
        "alphas": alphas,
        "localization": dataset["localization_method"],
        "locata_test_loss": loss_by_config if test_strategy == "locata_random_per_speaker" else None,
        "test_strategy": test_strategy,
        "locata_test_samples_per_speaker": int(
            evaluation.get("locata_static_test_samples_per_speaker", 10)
        ),
    }

    runner = YamlExperimentRunner(
        loss_by_config=loss_by_config,
        kmax=int(dataset["kmax"]),
        config=runner_config,
        free_risks=calibration["free_risks"],
    )
    final_results, miscoverage_instances, midetect_instances = runner.run()

    summary = {
        "mean_Loss_FA": final_results["Loss_FA"].mean(),
        "mean_Loss_Area": final_results["Loss_Area"].mean(),
        "count_Loss_MC_gt_alpha1": int((final_results["Loss_MC"] > alphas[1]).sum()),
        "count_Loss_MD_gt_alpha0": int((final_results["Loss_MD"] > alphas[0]).sum()),
        "MC_mean": final_results["Loss_MC"].mean(),
        "MD_mean": final_results["Loss_MD"].mean(),
        "mean_angular_err": (
            final_results[["azi_err", "ele_err"]].mean(axis=1).mean()
            if {"azi_err", "ele_err"}.issubset(final_results.columns)
            else np.nan
        ),
        "miscoverage_instances": miscoverage_instances,
        "midetect_instances": midetect_instances,
    }
    print_summary_table(
        summary,
        alphas,
        float(calibration["delta"]),
        int(simulation["num_iterations"]),
    )


if __name__ == "__main__":
    main()
