import os
from collections import deque

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm

from scipy.spatial.distance import cdist

from Code.plots import plot_roi_neighbours
from Code.utilities import create_save_directory, normalize

class CoverageSet:
    def __init__(self, true_position, estimated_positions, likelihood_maps, lambda_list, room, path_, plot_function):
        self.true_position = true_position # [sets, speakers, coordinates]
        self.estimated_positions = estimated_positions # [sets, speakers, coordinates]
        self.likelihood_maps = likelihood_maps # [lx, ly]
        self.room = room
        self.lambda_list = np.sort(lambda_list)[::-1]
        self.path = path_

        # initialize arrays to accumulate results
        self.calib_sets = self.likelihood_maps.shape[0]
        self._speakers = self.likelihood_maps.shape[1]

        self.coverage_per_lambda = np.zeros((self._speakers, len(self.lambda_list)))
        self.area_per_lambda = np.zeros((self._speakers, len(self.lambda_list)))

        self._correction_term = 1 / (self.calib_sets + 1) # B/(n+1)
        self._save_dir = None
        self._do_plot = None
        self._mis_cov_rate = None
        self._test_coverage = None
        self._test_area = None
        self.plot_function = plot_function

    def calibrate(self, plot: bool = False, plot_coverage_set=False):
        """
        For each estimated position, "grow" a coverage region around the corresponding cell until
        the region covers the entire ROI (i.e. full likelihood map) and then compute, for each lambda threshold,
        whether the true cell is inside the region and the corresponding area.
        """
        self._do_plot = plot

        for set_index in tqdm(range(self.calib_sets), desc='Growing coverage set'):

            if plot and plot_coverage_set:
                current_path = create_save_directory(os.path.join(self.path, f'set_{set_index}'))

            all_covered = np.zeros(self._speakers, dtype=bool)

            if self._do_plot:
                fig, axes = plt.subplots(1, self._speakers+1, figsize=(5 * self._speakers, 4))

            true_speaker_order, estimated_speaker_order = self._match_estimated_to_source(self.true_position[set_index],self.estimated_positions[set_index])
            for lambda_index, lambda_ in enumerate(self.lambda_list):

                if np.all(all_covered):
                    break

                for true_speaker, estimated_speaker in zip(true_speaker_order, estimated_speaker_order):

                    if all_covered[estimated_speaker]:
                        continue

                    likelihood_map = self.likelihood_maps[set_index, estimated_speaker, ...]
                    for i in range(self._speakers):
                        if i == estimated_speaker:
                            continue
                    likelihood_map = normalize(likelihood_map)

                    estimated_source_grid = self._project_onto_grid(self.estimated_positions[set_index, estimated_speaker])
                    true_source_grid = self._project_onto_grid(self.true_position[set_index, true_speaker])

                    coverage_set = self.neighbours_coverage_set(likelihood_map, lambda_, estimated_position=estimated_source_grid)

                    if coverage_set[true_source_grid[0], true_source_grid[1]]:
                        all_covered[estimated_speaker] = True
                        self.coverage_per_lambda[estimated_speaker, lambda_index:] += 1

                    if plot_coverage_set:
                        title = (f'Speaker: {estimated_speaker + 1}, '
                                 f'Lambda: {lambda_:.3f}, '
                                 f'Area: {np.sum(coverage_set) * self.room.resolution:.3f}, '
                                 f'Covered: {bool(coverage_set[true_source_grid[0], true_source_grid[1]])}')

                        if getattr(self.room, 'type', None) == 'sphere':
                            map_ = np.flipud(likelihood_map)
                        else:
                            map_ = likelihood_map

                        plot_roi_neighbours(likelihood_map=map_,
                                            mask=coverage_set,
                                            true_position=self.true_position[set_index, true_speaker],
                                            estimated_position=self.estimated_positions[set_index, estimated_speaker],
                                            room=self.room,
                                            title=title,
                                            figure_path=current_path)

                    # Plot only once. Lambda does not change the estimated position
                    if self._do_plot and lambda_index == 0:
                        self.plot_function(ax=axes[estimated_speaker],
                                           room=self.room,
                                           likelihood_map=likelihood_map,
                                           est_pos=self.estimated_positions[set_index, estimated_speaker],
                                           true_pos=self.true_position[set_index, true_speaker],
                                           order=estimated_speaker,
                                           title=f'Speaker {estimated_speaker+1}')


                        self.plot_function(ax=axes[-1],
                                           room=self.room,
                                           likelihood_map=np.maximum.reduce(self.likelihood_maps[set_index]),
                                           est_pos=self.estimated_positions[set_index, estimated_speaker],
                                           true_pos=self.true_position[set_index, true_speaker],
                                           order=estimated_speaker,
                                           title=f'Total Likelihood Map')

            if self._do_plot:
                figure_path = os.path.join(current_path, f'total_new_400_samples.png')
                fig.savefig(figure_path, dpi=300, bbox_inches='tight')
                plt.close(fig)

        self._mis_cov_rate = 1. - self.coverage_per_lambda / self.calib_sets

    def test(self, test_sets:int, estimated_positions, true_positions, likelihood_maps, significance_level: np.array, test_plot: bool = False):

        colors = ['Magenta', 'Magenta', 'Magenta', '']
        lineStyles = ['solid'] * self._speakers

        n = significance_level.shape[0]
        self._test_coverage = [[[] for _ in range(n)] for _ in range(self._speakers)]
        self._test_area = [[[] for _ in range(n)] for _ in range(self._speakers)]

        for err_ind, err in enumerate(significance_level):

            lambda_ = self._calc_conformal_risk_control(err)
            if lambda_.shape[0] == 1:
                lambda_ = np.repeat(lambda_, self._speakers)

            for lmbd_indx, lmbd in enumerate(lambda_):
                print(f'Conformal Risk Control: (error rate:{err}, Speaker: {lmbd_indx}: lambda:{lmbd:.3f})')

            for set_index in tqdm(range(test_sets), desc='Testing coverage set'):

                if test_plot:
                    fig, ax = plt.subplots(constrained_layout=True)
                    current_path = create_save_directory(os.path.join(self.path + '_test', f'set_{set_index}'))

                true_speaker_order, estimated_speaker_order = self._match_estimated_to_source(true_positions[set_index], estimated_positions[set_index])

                for estimated_speaker, true_speaker in zip(estimated_speaker_order, true_speaker_order):

                    likelihood_map = likelihood_maps[set_index, estimated_speaker, ...]
                    likelihood_map = normalize(likelihood_map)

                    estimated_source_grid = self._project_onto_grid(estimated_positions[set_index, estimated_speaker])
                    true_source_grid = self._project_onto_grid(true_positions[set_index, true_speaker])
                    coverage_set = self.neighbours_coverage_set(likelihood_map, lambda_[estimated_speaker], estimated_position=estimated_source_grid)
                    self._test_coverage[estimated_speaker][err_ind].append(coverage_set[true_source_grid[0], true_source_grid[1]])
                    self._test_area[estimated_speaker][err_ind].append(np.sum(coverage_set) * self.room.resolution)

                    map_to_plot = likelihood_maps[set_index, estimated_speaker_order[0], ...]
                    map_ = np.flipud(map_to_plot) if getattr(self.room, 'type', 'cartesian') == 'sphere' else map_to_plot

                    if test_plot:
                        plot_roi_neighbours(ax=ax,
                                            likelihood_map=map_,
                                            mask=coverage_set,
                                            true_position=true_positions[set_index, true_speaker],
                                            estimated_position=estimated_positions[set_index, estimated_speaker],
                                            room=self.room,
                                            segment_color=colors[estimated_speaker],
                                            linestyle = lineStyles[estimated_speaker],
                                            figure_path=current_path,
                                            speaker_index=estimated_speaker+1)

                if test_plot:
                    figure_path = os.path.join(current_path, f'coverage_{int((1-err)*100)}.png')
                    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    plot_roi_neighbours.legend_added = False

        return np.mean(np.array(self._test_coverage), axis=2), np.mean(np.array(self._test_area), axis=2)

    @staticmethod
    def get_patch(center_point, grid_shape, size=1):
        x, y = center_point
        x_indices = np.clip(np.arange(x - size, x + size + 1), 0, grid_shape[0] - 1)
        y_indices = np.clip(np.arange(y - size, y + size + 1), 0, grid_shape[1] - 1)
        return np.ix_(x_indices, y_indices)

    @staticmethod
    def _get_current_source(estimated, true):
        """
        Get the true source that is closest to the estimated source.
        :param estimated:
        :param true:
        :return:
        """
        current_source_index = np.argmin(np.linalg.norm(true - estimated, axis=1))
        current_source = true[current_source_index]
        return current_source

    @staticmethod
    def _match_estimated_to_source(true_pos, est_pos):
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

    def _project_onto_grid(self, position):
        """
        Project the real-world position onto the grid.
        :param position:
        :return:
        """

        x_projected = np.argmin(np.abs(position[0] - self.room.xl[:, 0]))
        y_projected = np.argmin(np.abs(position[1] - self.room.yl[0]))
        return np.array([x_projected, y_projected])

    def _calc_conformal_risk_control(self, err: int):
        """
        Calculate the lambda threshold for a given error rate.
        :param err: error rate
        :return: compatible lambda threshold
        """
        if not self._correction_term <= err:
            raise ValueError(f"Error rate {err} is too small. It should be greater than {self._correction_term}.")

        mask = list()
        speakers = self._mis_cov_rate.shape[0]
        for speaker in range(speakers):
            mask.append(self.calib_sets / (self.calib_sets + 1) * self._mis_cov_rate[speaker] + self._correction_term <= err)
        mask = np.array(mask)

        indices = [np.where(~row)[0].max()+1 if np.any(~row) else 0 for row in mask]

        return np.array([self.lambda_list[i] for i in indices])

    @staticmethod
    def neighbours_coverage_set(likelihood_map, threshold, estimated_position=None, extended_set=False):

        if estimated_position is None:
            estimated_position = np.unravel_index(np.argmax(likelihood_map), likelihood_map.shape)
        if not isinstance(estimated_position, tuple):
            estimated_position = tuple(estimated_position)

        # Create a boolean coverage_set to keep track of added elements.
        coverage_set = np.zeros_like(likelihood_map, dtype=bool)
        coverage_set[estimated_position] = True

        # Initialize a queue for BFS with the seed element.
        queue = deque([estimated_position])

        # Define offsets for the 4-connected neighbours (up, down, left, right).
        neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if extended_set:
            neighbor_offsets.extend([(-1, 1), (1, 1), (-1, -1), (-1, 1)])

        # Perform the region growing.
        while queue:
            i, j = queue.popleft()
            for di, dj in neighbor_offsets:
                ni, nj = i + di, j + dj
                # Check boundaries.
                if 0 <= ni < likelihood_map.shape[0] and 0 <= nj < likelihood_map.shape[1]:
                    # Add the neighbour if it hasn't been added and its value exceeds the threshold.
                    if not coverage_set[ni, nj] and likelihood_map[ni, nj] >= threshold:
                        coverage_set[ni, nj] = True
                        queue.append((ni, nj))

        coverage_set = binary_fill_holes(coverage_set)

        return coverage_set