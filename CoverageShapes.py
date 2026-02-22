from matplotlib import pyplot as plt, patches
import numpy as np
from collections import deque

from scipy.ndimage import binary_closing, binary_dilation, label, binary_fill_holes


class Square:
    def __init__(self, likelihood_map, boundary_x, boundary_y, LL, room_setup, grid_x, grid_y, speakers=1, reverb=700, snr=10):
        self.hop = 0
        self.LL = LL
        self.speakers = speakers
        self.mean = []
        self.room_setup = room_setup
        self.coverage_sets = []
        self.boundary_x, self.boundary_y = boundary_x - 1, boundary_y - 1
        self.likelihood_map = likelihood_map
        self._cal_mean = []
        self.cal_coverage_sets = []
        self.reverb = reverb
        self.snr = snr
        self.grid_x, self.grid_y = grid_x, grid_y
        self.region = None

        self._prepare_plot_setup()

    @property
    def calibration_mean(self):
        return np.array(self._cal_mean)

    def run(self, pos_x, pos_y):
        self.hop = 0
        row_min = row_max = col_min = col_max = None

        while not (row_min == 0 and row_max == self.boundary_x and col_min == 0 and col_max == self.boundary_y):
            row_min = max(pos_x - self.hop, 0)
            row_max = min(pos_x + self.hop, self.boundary_x)
            col_min = max(pos_y - self.hop, 0)
            col_max = min(pos_y + self.hop, self.boundary_y)

            region_top = self.likelihood_map[row_min, col_min:col_max + 1]
            region_bottom = self.likelihood_map[row_max, col_min:col_max + 1]
            region_left = self.likelihood_map[row_min + 1:row_max, col_min]
            region_right = self.likelihood_map[row_min + 1:row_max, col_max]

            mean = np.concatenate([region_top, region_right, region_bottom[::-1], region_left[::-1]]).mean()
            self._cal_mean.append(mean)
            self.cal_coverage_sets.append((row_min, row_max, col_min, col_max))

            self.hop += 1

        self.region = self.likelihood_map[row_min:row_max + 1, col_min:col_max + 1]

    def _prepare_plot_setup(self):
        self.axis_ticks = [self.grid_x[:, 0][0], self.grid_x[:, 0][-1], self.grid_y[0, :][0], self.grid_y[0, :][-1]]

        self.resolution_x = (self.axis_ticks[1] - self.axis_ticks[0]) / self.likelihood_map.shape[1]
        self.resolution_y = (self.axis_ticks[3] - self.axis_ticks[2]) / self.likelihood_map.shape[0]

    def _generate_roi(self, center):
        rect_width = 2 * (self.hop + 2) * self.resolution_x
        rect_height = 2 * (self.hop + 2) * self.resolution_y
        rect_x = center[0] - rect_width / 2
        rect_y = center[1] - rect_height / 2

        # Truncate width and height if the rectangle exceeds the boundaries
        if rect_x < self.axis_ticks[0]:
            rect_width -= (self.axis_ticks[0] - rect_x)
            rect_x = self.axis_ticks[0]
        if rect_y < self.axis_ticks[2]:
            rect_height -= (self.axis_ticks[2] - rect_y)
            rect_y = self.axis_ticks[2]
        if rect_x + rect_width > self.axis_ticks[1]:
            rect_width -= (rect_x + rect_width - self.axis_ticks[1])
        if rect_y + rect_height > self.axis_ticks[3]:
            rect_height -= (rect_y + rect_height - self.axis_ticks[3])

        return rect_x, rect_y, rect_width, rect_height

    def area_and_coverage(self, center, true_position):
        rect_x, rect_y, rect_width, rect_height = self._generate_roi(center)
        is_covered = (rect_x <= true_position[0] <= rect_x + rect_width and
                      rect_y <= true_position[1] <= rect_y + rect_height)

        return rect_width * rect_height, is_covered

    def plot_roi(self, true_point, estimated_point):
        rect_x, rect_y, rect_width, rect_height = self._generate_roi(estimated_point)

        fig, ax = plt.subplots()
        heatmap = ax.imshow(np.rot90(self.likelihood_map), extent=self.axis_ticks)
        fig.colorbar(heatmap, ax=ax)

        for i in range(self.speakers):
            ax.plot(
                estimated_point[i][0], estimated_point[i][1], '^',
                color='red', markersize=7,
                markeredgecolor='white', markerfacecolor='red'
            )
            ax.plot(
                true_point[0], true_point[1], '^',
                color='blue', markersize=7,
                markeredgecolor='white', markerfacecolor='blue',
            )

        rectangle = patches.Rectangle(
            (rect_x, rect_y), rect_width, rect_height,
            linewidth=1, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rectangle)

        for mic in self.room_setup:
            ax.plot(
                mic[0], mic[1], 'o',
                color='green', markersize=8,
                markeredgecolor='white', markerfacecolor='green'
            )

        ax.set_xlim(0, self.LL[0])
        ax.set_ylim(0, self.LL[1])
        # ax.set_title(title)
        ax.set_xlabel('X-Coordinate')
        ax.set_ylabel('Y-Coordinate')

        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

        ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.25, alpha=0.5)

        plt.show()
 

def neighbours_coverage_set(likelihood_map, threshold, estimated_position=None, extended_set=False, smooth=False):

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

    if smooth:
        # coverage_set += binary_closing(coverage_set, structure=np.ones((3, 3)))
        # labeled_matrix, num_components = label(coverage_set)
        # if num_components > 1:
        #     coverage_set = binary_dilation(coverage_set, structure=np.ones((2, 1)))
        coverage_set = binary_fill_holes(coverage_set)
    return coverage_set


def derivative_coverage_set(likelihood_map, threshold, use_diagonal=False):

    from scipy.ndimage import laplace

    rows, cols = likelihood_map.shape
    mask = np.zeros_like(likelihood_map, dtype=bool)

    # Compute gradients and Laplacian
    grad_y, grad_x = np.gradient(likelihood_map)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    # gradient_magnitude = (gradient_magnitude - np.min(gradient_magnitude)) / (np.max(gradient_magnitude) - np.min(gradient_magnitude))

    # Determine starting point
    start = np.unravel_index(np.argmax(likelihood_map), likelihood_map.shape)

    queue = deque([start])
    mask[start] = True

    # 4-connected neighbors
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        i, j = queue.popleft()

        for di, dj in neighbors:
            ni, nj = i + di, j + dj

            if 0 <= ni < rows and 0 <= nj < cols and not mask[ni, nj]:
                # Apply gradient and Laplacian constraints
                if gradient_magnitude[ni, nj] <= threshold:
                    mask[ni, nj] = True
                    queue.append((ni, nj))

    return mask


def plot_vector_field(map, grad_x, grad_y):
    X, Y = np.meshgrid(np.arange(map.shape[1]),
                       np.arange(map.shape[0]))

    # Plotting
    plt.figure(figsize=(8, 6))

    # Show the likelihood map as background
    plt.imshow(map, cmap='viridis', origin='upper')
    plt.colorbar(label='Likelihood')

    # Add the quiver plot (vector field)
    plt.quiver(X, Y, grad_x, -grad_y, color='white', angles='xy', scale_units='xy', scale=1.5)

    # Labels and title
    plt.title("Likelihood Map with Gradient Vector Field")
    plt.xlabel("X-axis (Columns)")
    plt.ylabel("Y-axis (Rows)")

    plt.show()
