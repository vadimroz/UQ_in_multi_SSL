import numpy as np

from Code.CoverageShapes import neighbours_coverage_set
from Code.utilities import normalize


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

def project_onto_grid(position, room):
    x_projected = np.argmin(np.abs(position[0] - room.xl[:, 0]))
    y_projected = np.argmin(np.abs(position[1] - room.yl[0]))
    return np.array([x_projected, y_projected])
