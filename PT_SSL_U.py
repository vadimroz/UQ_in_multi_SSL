import argparse
import itertools
from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm

from Code.PT_SSL_U.utilities import compute_risks, compute_p_values_wsr, compute_Pareto_frontier
from Code.utilities import *

seed = 1234567890
np.random.seed(seed)

def parse_arguments():
    """Parse CLI arguments for dataset selection and experiment hyperparameters."""
    parser = argparse.ArgumentParser(description="Run PT_SSL_U.py with configurable parameters.")
    parser.add_argument("--dataset", type=str, default="SYNTHETIC", help="Dataset type (SYNTHETIC or LOCATA).")
    parser.add_argument("--localization", type=str, default="SRP_PHAT", help="Localization method (SRP_PHAT or SRP_DNN).")
    parser.add_argument("--test_on_locata", type=bool, default=False, help="Test on LOCATA dataset.")
    parser.add_argument("--num_iterations", type=int, default=100, help="Number of iterations.")
    parser.add_argument("--calib_opt", type=int, default=200, help="Number of calibration optimization sets.")
    parser.add_argument("--calib_test", type=int, default=200, help="Number of calibration test sets.")
    parser.add_argument("--test_sets", type=int, default=100, help="Number of test sets.")
    parser.add_argument("--snr", type=int, default=15, help="Signal-to-noise ratio in dB.")
    parser.add_argument("--reverb", type=int, default=400, help="Reverberation time in ms.")
    parser.add_argument("--delta", type=float, default=0.1, help="Significance level.")
    parser.add_argument("--alpha_MC", type=float, default=0.1, help="MC tolerance level.")
    parser.add_argument("--alpha_MD", type=float, default=0.1, help="MD tolerance level.")
    return parser.parse_args()


@dataclass
class IterationSplit:
    """Per-iteration split for optimization calibration, test calibration, and held-out test."""
    cal_opt: dict
    cal_test: dict
    test: dict


@dataclass
class OptimizationResult:
    """Outputs of the optimization stage used by calibration."""
    lambda_pareto: pd.DataFrame
    pareto_combinations: np.ndarray
    lambda_ordered: list


class DataManager:
    """Centralized dataframe filtering helpers used across all engines."""
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
    """Build Pareto candidates from optimization calibration data."""
    def __init__(self, data_manager: DataManager, kmax: int, alphas: np.ndarray, delta: float, lambda_grid: list):
        self.data_manager = data_manager
        self.kmax = kmax
        self.alphas = alphas
        self.delta = delta
        self.lambda_grid = lambda_grid

    def run(self, split: IterationSplit) -> OptimizationResult:
        cal_opt_combined = self.data_manager.combine_rows(split.cal_opt)
        opt_risks = compute_risks(cal_opt_combined, self.kmax)

        costs = opt_risks.loc[:, opt_risks.columns.str.startswith("Risk")].to_numpy()
        efficient_indices = compute_Pareto_frontier(costs)
        lambda_pareto = opt_risks[efficient_indices]

        # Keep only configs on the constrained-risk Pareto frontier.
        pareto_combinations = np.flatnonzero(efficient_indices)
        pareto_set_losses = cal_opt_combined[cal_opt_combined["config_index"].isin(pareto_combinations)]
        p_values_opt_wsr = compute_p_values_wsr(
            pareto_set_losses,
            pareto_combinations,
            self.alphas,
            self.delta,
            self.kmax,
        )

        lambda_pareto = lambda_pareto.copy()
        lambda_pareto.loc[:, "p_values"] = p_values_opt_wsr
        lambda_pareto.sort_values(by="p_values", ascending=True, inplace=True)
        lambda_ordered = [self.lambda_grid[idx] for idx in lambda_pareto["config_index"].tolist()]

        return OptimizationResult(
            lambda_pareto=lambda_pareto,
            pareto_combinations=pareto_combinations,
            lambda_ordered=lambda_ordered,
        )


class CalibrationEngine:
    """Apply FST-based calibration and choose one configuration index."""
    def __init__(self, data_manager: DataManager, kmax: int, alphas: np.ndarray, delta: float, free_risks: list):
        self.data_manager = data_manager
        self.kmax = kmax
        self.alphas = alphas
        self.delta = delta
        self.free_risks = free_risks

    def select_configuration(self, split: IterationSplit, optimization_result: OptimizationResult) -> int:
        cal_test_combined = self.data_manager.combine_rows(split.cal_test)
        cal_test_combined = cal_test_combined[
            cal_test_combined["config_index"].isin(optimization_result.lambda_pareto["config_index"])
        ]
        cal_risks = compute_risks(cal_test_combined, self.kmax)

        pareto_set_losses = cal_test_combined[
            cal_test_combined["config_index"].isin(optimization_result.pareto_combinations)
        ]
        p_values_testing = compute_p_values_wsr(
            pareto_set_losses,
            optimization_result.pareto_combinations,
            self.alphas,
            self.delta,
            self.kmax,
        )

        cal_risks = cal_risks.copy()
        cal_risks.loc[:, "p_values"] = p_values_testing
        cal_risks.sort_values(by="p_values", ascending=True, inplace=True)

        # FST step: keep only non-rejected hypotheses.
        lambda_rejected = cal_risks[cal_risks["p_values"] <= self.delta]
        costs_free = lambda_rejected.loc[:, lambda_rejected.columns.isin(self.free_risks)].to_numpy()
        efficient_indices = compute_Pareto_frontier(costs_free)
        lambda_star = lambda_rejected[efficient_indices]

        try:
            return lambda_star.loc[lambda_star["Risk_Area"].idxmin()].filter(like="config_index").item()
        except Exception:
            return cal_risks.loc[cal_risks["p_values"].idxmin()].filter(like="config_index").item()


class BaseTestingEngine:
    """Shared testing post-processing; subclasses implement dataset-specific sampling."""
    def __init__(self, data_manager: DataManager, kmax: int, calib_opt: int):
        self.data_manager = data_manager
        self.kmax = kmax
        self.calib_opt = calib_opt

    def _build_test_set(self, split: IterationSplit, chosen_config_index: int) -> pd.DataFrame:
        raise NotImplementedError

    def evaluate(self, split: IterationSplit, chosen_config_index: int, iteration_index: int) -> pd.Series:
        test_combined = self._build_test_set(split, chosen_config_index)
        test_combined["Loss_Area"] = (
            test_combined["Loss_Area"] / test_combined["Estimated_speakers"].replace(0, np.nan)
        ).fillna(0)
        test_combined = test_combined.tail(297).reset_index(drop=True)

        res = test_combined[["Loss_MC", "Loss_MD", "Loss_FA", "Loss_Area"]].mean()
        res["Iteration"] = iteration_index
        res["n_calibration"] = 2 * self.calib_opt
        return res


class LocataTestingEngine(BaseTestingEngine):
    """Testing strategy for LOCATA evaluation protocol."""
    def __init__(self, data_manager: DataManager, kmax: int, calib_opt: int, locata_loss: pd.DataFrame):
        super().__init__(data_manager, kmax, calib_opt)
        self.locata_loss = locata_loss

    def _build_test_set(self, split: IterationSplit, chosen_config_index: int) -> pd.DataFrame:
        test_combined = self.locata_loss[self.locata_loss["config_index"] == chosen_config_index]
        speaker_2 = test_combined[test_combined["True_speakers"] == 2].sample(n=10, replace=False)
        speaker_1 = test_combined[test_combined["True_speakers"] == 1].sample(n=10, replace=False)
        return pd.concat([speaker_1, speaker_2], axis=0)


class SyntheticTestingEngine(BaseTestingEngine):
    """Testing strategy for synthetic held-out folds."""
    def _build_test_set(self, split: IterationSplit, chosen_config_index: int) -> pd.DataFrame:
        return self.data_manager.combine_rows(split.test, config_index=chosen_config_index)


class ExperimentRunner:
    """Orchestrate optimization, calibration, and testing over all iterations."""
    def __init__(self, loss_by_config: pd.DataFrame, kmax: int, config: dict):
        self.loss_by_config = loss_by_config
        self.kmax = kmax
        self.config = config
        self.data_manager = DataManager(loss_by_config=loss_by_config, kmax=kmax)

    def _build_lambda_grid(self):
        mc1_grid = self.loss_by_config["Threshold_MC_1"].unique().tolist()
        mc2_grid = self.loss_by_config["Threshold_MC_2"].unique().tolist()
        if self.kmax >= 3:
            mc3_grid = self.loss_by_config["Threshold_MC_3"].unique().tolist()
            mc_grids = [mc1_grid, mc2_grid, mc3_grid]
        else:
            mc_grids = [mc1_grid, mc2_grid]
        md_grid = self.loss_by_config["Threshold_MD"].unique().tolist()

        mc_combinations = list(itertools.product(*mc_grids))
        return [(mc_vals, md_val) for mc_vals in mc_combinations for md_val in md_grid]

    def _build_iteration_split(self, iteration_folds) -> IterationSplit:
        # Split each speaker calibration set into optimization and calibration-test partitions.
        cal_opt = {}
        cal_test = {}
        test = {}

        for speaker_id in range(1, self.kmax + 1):
            speaker_cal, speaker_test = iteration_folds[speaker_id - 1]
            speaker_cal_opt = np.sort(speaker_cal[: self.config["calib_opt"]])[: self.config["calib_opt"]]
            speaker_cal_test = np.sort(speaker_cal[self.config["calib_opt"] :])[: self.config["calib_opt"]]

            cal_opt[speaker_id] = speaker_cal_opt
            cal_test[speaker_id] = speaker_cal_test
            test[speaker_id] = speaker_test

        return IterationSplit(cal_opt=cal_opt, cal_test=cal_test, test=test)

    def run(self):
        # Build engines once and reuse across iterations.
        lambda_grid = self._build_lambda_grid()
        splits = generate_random_splits(
            total_samples=self.config["samples"],
            num_iterations=self.config["num_iterations"],
            calib_size=int(self.config["samples"] * 0.8),
            num_lists=self.kmax,
        )
        folds_across_lists = list(zip(*splits))

        optimization_engine = OptimizationEngine(
            data_manager=self.data_manager,
            kmax=self.kmax,
            alphas=self.config["alphas"],
            delta=self.config["delta"],
            lambda_grid=lambda_grid,
        )
        calibration_engine = CalibrationEngine(
            data_manager=self.data_manager,
            kmax=self.kmax,
            alphas=self.config["alphas"],
            delta=self.config["delta"],
            free_risks=["Risk_Area", "Risk_FA"],
        )
        # Choose testing strategy based on target evaluation dataset.
        if self.config["test_on_locata"]:
            locata_dataset = f"data/{self.config['localization']}/LOCATA/locata_loss.parquet"
            locata_loss = pd.read_parquet(locata_dataset, engine="pyarrow")

            testing_engine = LocataTestingEngine(
                data_manager=self.data_manager,
                kmax=self.kmax,
                calib_opt=self.config["calib_opt"],
                locata_loss=locata_loss,
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

        for iteration_index in tqdm(range(self.config["num_iterations"]), desc="Running iterations", unit="iter"):
            split = self._build_iteration_split(folds_across_lists[iteration_index])

            optimization_result = optimization_engine.run(split)
            chosen_config_index = calibration_engine.select_configuration(split, optimization_result)
            res = testing_engine.evaluate(split, chosen_config_index, iteration_index)
            test_results.append(res)

            if res["Loss_MC"] > self.config["alphas"][1]:
                miscoverage_instances += 1
            if res["Loss_MD"] > self.config["alphas"][0]:
                midetect_instances += 1

            print(
                f"Iteration {iteration_index + 1}/{self.config['num_iterations']} - "
                f"Res: {res.to_dict()}"
            )

        return pd.DataFrame(test_results), miscoverage_instances, midetect_instances


def main():
    """Entry point: load data, run experiment, and print summary metrics."""

    kmax = 3
    args = parse_arguments()

    dataset = args.dataset
    localization = args.localization
    calib_opt = args.calib_opt
    calib_test = args.calib_test
    calibration_sets = calib_test + calib_opt
    test_sets = args.test_sets
    samples = calibration_sets + test_sets
    alpha_mc = args.alpha_MC
    alpha_md = args.alpha_MD
    alphas = np.array([alpha_md, alpha_mc])  # [MD, MC]

    assert alpha_mc == alpha_md, "alpha_MC and alpha_MD must be same."
    assert kmax <= 3, "Kmax greater than 3 is not supported by current data loading logic."

    if args.test_on_locata:
        kmax = 2
        dataset = "LOCATA_Matched"

        # LOCATA dataset room conditions
        args.snr = 20 #dB
        args.reverb = 550 #ms

        print(f"Testing on LOCATA with Kmax= 2, SNR={args.snr} dB and Reverb={args.reverb} ms ...")

    all_risks_filename = (
        f"data/{localization}/{dataset}/Reverb_{args.reverb}_ms_SNR_{args.snr}_dB/dataset.parquet"
    )

    print(f"Loading dataset from {all_risks_filename} ...")
    loss_by_config = pd.read_parquet(all_risks_filename, engine="pyarrow")

    runner_config = {
        "test_on_locata": args.test_on_locata,
        "num_iterations": args.num_iterations,
        "calib_opt": calib_opt,
        "samples": samples,
        "delta": args.delta,
        "alphas": alphas,
        "localization": localization,
    }
    runner = ExperimentRunner(loss_by_config=loss_by_config, kmax=kmax, config=runner_config)
    final_results, miscoverage_instances, midetect_instances = runner.run()

    summary = {
        "mean_Loss_FA": final_results["Loss_FA"].mean(),
        "mean_Loss_Area": final_results["Loss_Area"].mean(),
        "count_Loss_MC_gt_alpha1": int((final_results["Loss_MC"] > alphas[1]).sum()),
        "count_Loss_MD_gt_alpha0": int((final_results["Loss_MD"] > alphas[0]).sum()),
        "MC_mean": final_results["Loss_MC"].mean(),
        "MD_mean": final_results["Loss_MD"].mean(),
        "miscoverage_instances": miscoverage_instances,
        "midetect_instances": midetect_instances,
    }
    print(summary)


if __name__ == "__main__":
    main()
