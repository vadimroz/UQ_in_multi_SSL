import os
from datetime import datetime

import numpy as np
from tabulate import tabulate
from sklearn.model_selection import train_test_split

class LocalizationMethod:
    SRP_PHAT = 'srp_phat'
    SRP_DNN = 'srp_dnn'

class DatasetType:
    LOCATA_MATCHED = 'locata_matched'
    LOCATA_AUG = 'locata_augmented'
    SYNTHETIC = 'Synthetic'
    SYNTETHIC_MANY_ROOMS = 'synthetic_many_rooms'

def normalize(x, a=0, b=1):
    x_min = np.min(x)
    x_max = np.max(x)
    return a + (x - x_min) * (b - a) / (x_max - x_min)

def create_save_directory(dir_name: str):
    cwd = os.getcwd()
    if not dir_name:
        dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(cwd, dir_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def compute_softmax_probability(P_map, theta_idx, window_radius, alpha=1.0):
    """
    Compute softmax probability Pr(theta_k) for a given detected theta_k.

    Args:
        P_map (np.ndarray): SRP-PHAT map, shape = (num_candidates,)
        theta_idx (int): index of the detected direction theta_k
        alpha (float): sharpness scaling factor (temperature)

    Returns:
        float: Probability assigned to theta_k
    """
    # Stabilize exponentials to avoid overflow
    # P_map_scaled = P_map_scaled #- np.max(P_map_scaled)  # Subtract max for numerical stability

    x, y = theta_idx
    xmin, xmax = max(x - window_radius, 0), min(x + window_radius + 1, P_map.shape[0])
    ymin, ymax = max(y - window_radius, 0), min(y + window_radius + 1, P_map.shape[1])

    P_map_scaled = alpha * P_map
    exp_P = np.exp(P_map_scaled)

    local_patch = exp_P[xmin:xmax, ymin:ymax]
    softmax_sum = np.sum(local_patch)
    prob_theta_k = exp_P[theta_idx] / softmax_sum

    return prob_theta_k

def print_results(
    miscoverage_array,
    area_array,
    mean_angular_error=None,
    speakers=None,
    significance_level=(0.1, 0.05, 0.01),
    tablefmt="rounded_outline",
    area_unit="",
    title="CRC evaluation summary",
):
    def as_result_matrix(values, name):
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 0:
            return arr.reshape(1, 1)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        if arr.ndim == 2:
            return arr
        raise ValueError(f"{name} must be scalar, 1D, or 2D.")

    miscov = as_result_matrix(miscoverage_array, "miscoverage_array")
    area = as_result_matrix(area_array, "area_array")
    angular_error = (
        as_result_matrix(mean_angular_error, "mean_angular_error")
        if mean_angular_error is not None
        else None
    )
    sig = np.atleast_1d(np.asarray(significance_level, dtype=float))

    if miscov.shape != area.shape or (
        angular_error is not None and angular_error.shape != area.shape
    ):
        raise ValueError(
            "Shape mismatch: "
            f"miscoverage_array {miscov.shape}, area_array {area.shape}, "
            f"mean_angular_error "
            f"{None if angular_error is None else angular_error.shape}."
        )
    if miscov.shape[1] != sig.size:
        raise ValueError(
            f"Number of significance levels ({sig.size}) must match the second dimension "
            f"of the arrays ({miscov.shape[1]})."
        )

    S, L = miscov.shape
    if speakers is None:
        speakers = S
    if speakers != S:
        raise ValueError(f"'speakers' ({speakers}) does not match number of rows ({S}).")

    headers = ["Speaker", "Alpha", "Target coverage", "Miscoverage", f"Area [{area_unit}]"]
    if angular_error is not None:
        headers.append("Mean angular error [deg]")

    rows = []
    for i in range(S):
        for j, alpha in enumerate(sig):
            row = [
                f"Speaker {i + 1}",
                f"{alpha:.3f}",
                f"{1.0 - alpha:.1%}",
                f"{miscov[i, j]:.2%}",
                f"{area[i, j]:.3f}",
            ]
            if angular_error is not None:
                row.append(f"{angular_error[i, j]:.3f}")
            rows.append(row)

    print(f"\n{title}")
    print(tabulate(rows, headers=headers, tablefmt=tablefmt))


def flatten_dict(dicts, key):
    max_rows = max(d[key].shape[0] for d in dicts)
    return sum(np.pad(d[key], ((0, max_rows - d[key].shape[0]), (0, 0))) for d in dicts)


def aggregate_risks(risks_by_speaker):
    """
    Aggregate risks by taking the mean of each risk type for each speaker.
    """
    res_dict = {}
    all_far = []
    all_mdr = []
    all_areas = []
    for k, v in sorted(risks_by_speaker.items()):
        res_dict[f'MC_{k}'] = np.mean(v['mc'])
        all_mdr.extend(v['mdr'])
        all_far.extend(v['far'])
        all_areas.extend(v['area'])

    res_dict['area_mean'] = np.mean(all_areas)
    res_dict['far_mean'] = np.mean(all_far)
    res_dict['mdr_mean'] = np.mean(all_mdr)

    res_array = np.array(list(res_dict.values()))
    return res_dict, res_array


from collections import defaultdict
from pprint import pprint


def print_summary(data):
    # Group data by ground_truth_speakers
    summary = defaultdict(lambda: {"mc": 0, "mdr": 0, "area": 0, "far": 0})

    for entry in data:
        speakers = entry["ground_truth_speakers"]
        summary[speakers]["mc"] += entry["mc"]
        summary[speakers]["mdr"] += entry["mdr"]
        summary[speakers]["area"] += entry["area"]
        summary[speakers]["far"] += entry["far"]

    # Pretty print the summary
    pprint(dict(summary))

def print_average_risks(data):
    # Group data by ground_truth_speakers
    grouped_risks = defaultdict(lambda: {"mc": [], "mdr": [], "area": [], "far": []})

    for entry in data:
        speakers = entry["ground_truth_speakers"]
        grouped_risks[speakers]["mc"].append(entry["mc"])
        grouped_risks[speakers]["mdr"].append(entry["mdr"])
        grouped_risks[speakers]["area"].append(entry["area"])
        grouped_risks[speakers]["far"].append(entry["far"])

    # Compute averages for each group
    averages = {
        speakers: {
            "mc": sum(values["mc"]) / len(values["mc"]),
            "mdr": sum(values["mdr"]) / len(values["mdr"]),
            "area": sum(values["area"]) / len(values["area"]),
            "far": sum(values["far"]) / len(values["far"]),
        }
        for speakers, values in grouped_risks.items()
    }

    return pprint(averages)

def print_risks_by_speakers(data):
    # Group data by ground_truth_speakers
    grouped_risks = defaultdict(list)

    for entry in data:
        speakers = entry["ground_truth_speakers"]
        grouped_risks[speakers].append({
            "mc": entry["mc"],
            "mdr": entry["mdr"],
            "area": entry["area"],
            "far": entry["far"]
        })

    return pprint(dict(grouped_risks))

def dominates(a, b):
    return np.all(a <= b, axis=-1) & np.any(a < b, axis=-1)

def zero_likelihood_rectangle(likelihood_map, ele_max, azi_max, radius=3):

    height, width = likelihood_map.shape

    y_min = max(0, ele_max - radius)
    y_max = min(height, ele_max + radius + 1)
    x_min = max(0, azi_max - radius)
    x_max = min(width, azi_max + radius + 1)

    likelihood_map[y_min:y_max, x_min:x_max] = 0

    return likelihood_map

def cart2sph(cart):
	""" cart [x,y,z] → sph [r,ele,azi]
	"""
	xy2 = cart[:,0]**2 + cart[:,1]**2
	sph = np.zeros_like(cart)
	sph[:,0] = np.sqrt(xy2 + cart[:,2]**2)
	sph[:,1] = np.arctan2(np.sqrt(xy2), cart[:,2]) # Elevation angle defined from Z-axis down
	sph[:,2] = np.arctan2(cart[:,1], cart[:,0])
	return sph

def angular_distance(pos1, pos2):
    """
    Compute angular distance between two positions in elevation and azimuth
    pos1, pos2: [elevation, azimuth]
    """
    d_ele = abs(np.degrees(pos1[0] - pos2[0]))
    d_azi = abs(np.degrees(pos1[1] - pos2[1]))
    d_azi = min(d_azi, 360 - d_azi)  # Handle wrap-around for azimuth
    return d_ele, d_azi

def generate_random_splits(total_samples=500, num_iterations=5, calib_size=400, num_lists=3, random_seed=42):
    """
    Generate multiple independent lists of calibration/test splits.

    Returns:
        A list of lists:
            [
                [ (calib_index_iter1, test_index_iter1), ... ],  # List 1
                [ (calib_index_iter1, test_index_iter1), ... ],  # List 2
                [ (calib_index_iter1, test_index_iter1), ... ]   # List 3
            ]
    """
    assert calib_size < total_samples, "Calibration size must be less than total samples"

    all_splits = []

    for list_idx in range(num_lists):
        splits = []
        for i in range(num_iterations):
            calib_index, test_index = train_test_split(
                np.arange(total_samples),
                train_size=calib_size,
                shuffle=True,
                random_state=random_seed + list_idx * 1000 + i
            )
            splits.append((calib_index, test_index))
        all_splits.append(splits)

    return all_splits

def calc_angular_error(pos_true, post_est):
    ele_true, azi_true = pos_true
    ele_est, azi_est = post_est

    u_true = np.array([np.sin(ele_true) * np.cos(azi_true), np.sin(ele_true) * np.sin(azi_true), np.cos(ele_true)])
    u_est = np.array([np.sin(ele_est) * np.cos(azi_est), np.sin(ele_est) * np.sin(azi_est), np.cos(ele_est)])

    dot_product = np.clip(np.dot(u_true, u_est), -1.0, 1.0)
    angular_error = np.rad2deg(np.arccos(dot_product))

    return angular_error

def calc_azi_ele_err_degrees(pos_true, post_est):
    ele_true, azi_true = np.rad2deg(pos_true)
    ele_est, azi_est = np.rad2deg(post_est)

    ele_err = np.abs(ele_true - ele_est)
    azi_err = np.abs((azi_true - azi_est + 180) % 360 - 180)

    return azi_err, ele_err


def get_neighborhood(arr, x, y, shift_left, shift_right):
    return arr[y - shift_left: y + shift_right, x - shift_left: x + shift_right]


def local_sharpness(arr, y, x, shift_left=1, shift_right=2):
    center = arr[y, x]
    neighborhood = get_neighborhood(arr, x=x, y=y, shift_left=shift_left, shift_right=shift_right)
    total_neighbors = np.sum(neighborhood) - center
    weight = neighborhood.size - 1
    sharpness = (weight * center) - total_neighbors
    return sharpness, neighborhood
