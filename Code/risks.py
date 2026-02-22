import numpy as np
from scipy.spatial.distance import cdist

from Code.CoverageShapes import neighbours_coverage_set
from scipy.ndimage import label, binary_fill_holes

from Code.utilities import normalize


def mdr_far_risks(Kmax, softmax, ground_truth_speakers, threshold):
    detection_mask = softmax[:ground_truth_speakers] >= threshold

    # When md occurs, all future detections are also considered as mis-detections since iterative detection is used
    detection_mask = np.cumprod(detection_mask, axis=0).astype(bool)
    detected_speakers = np.sum(detection_mask, axis=0)
    detect_rate = detected_speakers / ground_truth_speakers
    mis_detect_rate = 1 - detect_rate

    spurious_speakers = 0
    false_alarm_rate = 0
    if detected_speakers == ground_truth_speakers:
        mask_false_alarm = softmax[ground_truth_speakers:] >= threshold
        mask_false_alarm = np.cumprod(mask_false_alarm, axis=0).astype(bool)
        spurious_speakers = np.sum(mask_false_alarm, axis=0)
        if Kmax > ground_truth_speakers:
            false_alarm_rate = spurious_speakers / (Kmax - ground_truth_speakers)
        else:
            false_alarm_rate = 0

    estimated_speakers = detected_speakers + spurious_speakers

    return mis_detect_rate, false_alarm_rate, estimated_speakers

def md_fa_risks(softmax, ground_truth_speakers, threshold):
    false_alarm = 0

    detection_mask = softmax[:ground_truth_speakers] >= threshold

    # When md occurs, all future detections are also considered as mis-detections since iterative detection is used
    detection_mask = np.cumprod(detection_mask, axis=0).astype(bool)
    detected_speakers = np.sum(detection_mask, axis=0)

    if detected_speakers == ground_truth_speakers:
        mask_false_alarm = softmax[ground_truth_speakers:] >= threshold
        mask_false_alarm = np.cumprod(mask_false_alarm, axis=0).astype(bool)
        false_alarm = np.sum(mask_false_alarm, axis=0)

    estimated_speakers = detected_speakers + false_alarm
    mis_detect = ground_truth_speakers - detected_speakers

    return mis_detect, false_alarm, estimated_speakers

def mc_area_risk(true_speakers, estimated_speakers, true_pos, estimated_pos, likelihood_maps, threshold, room):
    area = 0.0
    mis_coverage = np.min([estimated_speakers, true_speakers])

    if true_speakers < estimated_speakers:
        dummy_locations = np.full((estimated_speakers-true_speakers, true_pos.shape[1]), np.inf)
        true_pos = np.concatenate([true_pos, dummy_locations], axis=0)

    true_source_grid = [project_onto_grid(true_pos[i], room) for i in range(true_speakers)]

    for i, est_pair in enumerate(estimated_pos):
        likelihood_map = normalize(likelihood_maps[i, ...]) # worked for locata dnn** 0.5

        estimated_source_grid = project_onto_grid(est_pair, room)

        coverage_set = neighbours_coverage_set(likelihood_map=likelihood_map,threshold=threshold[i],
                                               estimated_position=estimated_source_grid, smooth=True)

        for j, grid in enumerate(true_source_grid):
            if coverage_set[grid[0], grid[1]]:
                true_source_grid.pop(j)
                mis_coverage -= 1
                break

        area += np.sum(coverage_set) * room.resolution

    return mis_coverage, area

def mc_area_risk_speakers_known(true_speakers, true_pos, estimated_pos, likelihood_maps, threshold, room):
    area = np.zeros(true_speakers)
    mis_coverage = np.ones(true_speakers)

    true_source_grid = [project_onto_grid(true_pos[i], room) for i in range(true_speakers)]

    for i, est_pair in enumerate(estimated_pos):
        likelihood_map = normalize(likelihood_maps[i, ...])

        estimated_source_grid = project_onto_grid(est_pair, room)

        coverage_set = neighbours_coverage_set(likelihood_map=likelihood_map,threshold=threshold[i],
                                               estimated_position=estimated_source_grid, smooth=True)

        for j, grid in enumerate(true_source_grid):
            if coverage_set[grid[0], grid[1]]:
                true_source_grid.pop(j)
                mis_coverage[i] -= 1
                break

        area[i] += np.sum(coverage_set) * room.resolution

    return mis_coverage, area

def match_estimated_to_source(true_pos, est_pos):
    true_pos = np.array(true_pos)
    est_pos = np.array(est_pos)

    # Distance matrix: true_pos x est_pos
    cost_matrix = cdist(true_pos, est_pos, metric='euclidean')

    assigned_true = set()
    true_indices = []
    est_indices = []

    for est_idx in range(est_pos.shape[0]):
        # Find the closest unassigned true_pos
        unassigned = [i for i in range(true_pos.shape[0]) if i not in assigned_true]
        if not unassigned:
            break
        sub_costs = cost_matrix[unassigned, est_idx]
        best_idx = unassigned[np.argmin(sub_costs)]
        assigned_true.add(best_idx)

        true_indices.append(best_idx)
        est_indices.append(est_idx)

    return np.array(true_indices), np.array(est_indices)

def assign_est_to_true(true_pos, est_pos):
    true_pos = np.array(true_pos)
    est_pos = np.array(est_pos)

    # Distance matrix: true x est
    cost_matrix = cdist(true_pos, est_pos, metric='euclidean')

    assigned_est = set()
    true_indices = []
    est_indices = []

    for true_idx in range(true_pos.shape[0]):
        # Find closest unassigned estimated point
        unassigned = [j for j in range(est_pos.shape[0]) if j not in assigned_est]
        if not unassigned:
            break
        sub_costs = cost_matrix[true_idx, unassigned]
        best_idx = unassigned[np.argmin(sub_costs)]
        assigned_est.add(best_idx)

        true_indices.append(true_idx)
        est_indices.append(best_idx)

    return np.array(true_indices), np.array(est_indices)

def src_est_matching(true_pos, est_pos):
    true_pos = np.array(true_pos)
    est_pos = np.array(est_pos)

    # Distance matrix: true_pos x est_pos
    cost_matrix = cdist(true_pos, est_pos, metric='euclidean')

    assigned_true = set()
    true_indices = []
    est_indices = []

    for est_idx in range(est_pos.shape[0]):
        # Find the closest unassigned true_pos
        unassigned = [i for i in range(true_pos.shape[0]) if i not in assigned_true]
        if not unassigned:
            break
        sub_costs = cost_matrix[unassigned, est_idx]
        best_idx = unassigned[np.argmin(sub_costs)]
        assigned_true.add(best_idx)

        true_indices.append(best_idx)
        est_indices.append(est_idx)

    return np.array(true_indices), np.array(est_indices)

def project_onto_grid(position, room):
    x_projected = np.argmin(np.abs(position[0] - room.xl[:, 0]))
    y_projected = np.argmin(np.abs(position[1] - room.yl[0]))
    return np.array([x_projected, y_projected])


def neighbours_coverage_set_fast(likelihood_map, threshold, estimated_position=None, extended_set=False, smooth=False):
    if estimated_position is None:
        estimated_position = np.unravel_index(np.argmax(likelihood_map), likelihood_map.shape)
    if not isinstance(estimated_position, tuple):
        estimated_position = tuple(estimated_position)

    # Create thresholded binary mask
    mask = likelihood_map >= threshold

    # Label connected components (4- or 8-connected)
    structure = np.array([[0,1,0],[1,1,1],[0,1,0]])
    labeled, num_features = label(mask, structure=structure)

    # Identify the label of the region containing the estimated position
    region_label = labeled[estimated_position]
    coverage_set = labeled == region_label

    if smooth:
        coverage_set = binary_fill_holes(coverage_set)

    return coverage_set
