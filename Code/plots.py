import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patheffects import withStroke
from matplotlib.ticker import MultipleLocator, FuncFormatter


def plot_roi_neighbours(likelihood_map, mask, true_position, estimated_position, room, title='', figure_path='', ax=None, segment_color='red', linestyle='solid', speaker_index= None):
    ax_empty = ax

    y_label = getattr(room, 'x_label', 'X-Coordinate')
    x_label = getattr(room, 'y_label', 'Y-Coordinate')

    if hasattr(room, 'extent'):
        extent = room.extent
    else:
        x_ticks, y_ticks = room.axis_ticks[:2], room.axis_ticks[2:]
        extent = [y_ticks[0], y_ticks[1], x_ticks[1], x_ticks[0]]

    if not hasattr(plot_roi_neighbours, "counter"):
        plot_roi_neighbours.counter = 1

    if not hasattr(plot_roi_neighbours, "legend_added"):
        plot_roi_neighbours.legend_added = False

    if ax_empty is None:
        fig, ax = plt.subplots(constrained_layout=True)
    else:
        fig = ax.get_figure()

    # Plot the speakers.

    ax.plot(
        estimated_position[1],
        estimated_position[0], 'x',
        color='red', markersize=5.5,
        markeredgecolor='red', markerfacecolor='red',
        label='Detected' if not plot_roi_neighbours.legend_added else None
    )

    ax.plot(
        true_position[1],
        true_position[0], 'x ',
        color='blue', markersize=5.5,
        markeredgecolor='blue', markerfacecolor='blue',
        label='True' if not plot_roi_neighbours.legend_added else None
    )

    if speaker_index:
        ax.text(
            estimated_position[1] + 0.15,
            estimated_position[0] + 0.15,
            speaker_index,
            color='red',
            fontsize=8,
            path_effects=[withStroke(linewidth=3, foreground='white')]
        )

    # # Plot the microphones.
    if hasattr(room, 'rp'):
        for mic_indx, mic in enumerate(room.rp):
            ax.plot(
                mic[1], mic[0], 'o',
                color='green', markersize=8,
                markeredgecolor='black', markerfacecolor='green',
                label='Microphone' if (not plot_roi_neighbours.legend_added and mic_indx==0) else None
            )

    # Set axis limits and labels.
    if hasattr(room, 'LL'):
        ax.set_xlim(0, room.LL[1])
        ax.set_ylim(0, room.LL[0])

    heatmap = ax.imshow(likelihood_map, extent=extent, interpolation='bilinear')

    if ax_empty is None:
        fig.colorbar(heatmap, ax=ax)

    ax.set_title(title)
    if getattr(room, 'type', None) == 'sphere':
        ax.set_xlabel(y_label)
        ax.set_ylabel(x_label)
        ax.xaxis.set_major_locator(MultipleLocator(np.pi / 2))
        ax.yaxis.set_major_locator(MultipleLocator(np.pi / 4))
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda val, pos: r"$\pi$" if round(val / np.pi, 2) == 1 else
            r"$-\pi$" if round(val / np.pi, 2) == -1 else
            r"${0}\pi$".format(round(val / np.pi, 2)) if val != 0 else "0"
        ))
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda val, pos: r"$\pi$" if round(val / np.pi, 2) == 1 else
            r"$-\pi$" if round(val / np.pi, 2) == -1 else
            r"${0}\pi$".format(round(val / np.pi, 2)) if val != 0 else "0"
        ))
    else:
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Optionally, set the visibility of spines and grid.
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    mask_ = mask if getattr(room, 'type', None) == 'sphere' else np.flip(mask, axis=0)
    segments = extract_grid_boundaries(mask_, extent)
    #
    # # Plot each segment.
    for seg in segments:
        (x1, y1), (x2, y2) = seg
        ax.plot([x1, x2], [y1, y2], segment_color, linewidth=1.75, linestyle=linestyle)

    if ax_empty is None:
        figure_path = os.path.join(figure_path, f'figure_{plot_roi_neighbours.counter}.png')
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()
        plot_roi_neighbours.counter += 1

    # if not plot_roi_neighbours.legend_added and any(line.get_label() != "_nolegend_" for line in ax.lines):
    #     ax.legend(loc='upper right')
    #     plot_roi_neighbours.legend_added = True

    if not getattr(room, 'type', None) == 'sphere':
        ax.set_xlim(x_ticks)
        ax.set_ylim(y_ticks)

def extract_grid_boundaries(mask, extent):
    """
    Extracts grid cell boundary segments for the True region in a binary mask.
    The returned segments follow exactly the grid lines ("legal" paths).

    Parameters:
      mask: 2D boolean numpy array.
      extent: [xmin, xmax, ymin, ymax] corresponding to the grid.

    Returns:
      segments: A list of segments. Each segment is a tuple of endpoints: ((x1, y1), (x2, y2)).
    """
    segments = []
    M, N = mask.shape  # rows, columns
    xmin, xmax, ymin, ymax = extent
    # Compute cell size
    dx = (xmax - xmin) / N
    dy = (ymax - ymin) / M

    # Loop over each cell in the mask
    for i in range(M):
        for j in range(N):
            if mask[i, j]:
                # For each True cell, check the 4 neighbors.
                # Top edge: if it's the first row or the cell above is False.
                if i == 0 or not mask[i - 1, j]:
                    # Top edge from (j, i) to (j+1, i)
                    x1 = xmin + j * dx
                    x2 = xmin + (j + 1) * dx
                    y_edge = ymin + i * dy
                    segments.append(((x1, y_edge), (x2, y_edge)))
                # Bottom edge: if it's the last row or the cell below is False.
                if i == M - 1 or not mask[i + 1, j]:
                    x1 = xmin + j * dx
                    x2 = xmin + (j + 1) * dx
                    y_edge = ymin + (i + 1) * dy
                    segments.append(((x1, y_edge), (x2, y_edge)))
                # Left edge: if it's the first column or the cell to the left is False.
                if j == 0 or not mask[i, j - 1]:
                    y1 = ymin + i * dy
                    y2 = ymin + (i + 1) * dy
                    x_edge = xmin + j * dx
                    segments.append(((x_edge, y1), (x_edge, y2)))
                # Right edge: if it's the last column or the cell to the right is False.
                if j == N - 1 or not mask[i, j + 1]:
                    y1 = ymin + i * dy
                    y2 = ymin + (i + 1) * dy
                    x_edge = xmin + (j + 1) * dx
                    segments.append(((x_edge, y1), (x_edge, y2)))
    return segments


def plot_roi_new(ax, room, likelihood_map, est_pos, true_pos, order, title=''):
    x_ticks, y_ticks = room.axis_ticks[:2],  room.axis_ticks[2:]
    extent = [y_ticks[0], y_ticks[1], x_ticks[1], x_ticks[0]]

    if est_pos is not None:
        ax.plot(est_pos[1], est_pos[0], '^',
                color='red',
                markersize=7,
                markeredgecolor='white',
                markerfacecolor='red')
        if order is not None:
            ax.text(est_pos[1] + 0.05, est_pos[0] + 0.05, order+1, fontsize=10, color='red')

    if true_pos is not None:
        ax.plot(true_pos[1], true_pos[0], '^',
                color='blue',
                markersize=7,
                markeredgecolor='white',
                markerfacecolor='blue')

    for mic in room.rp:
        ax.plot(
            mic[1], mic[0], 'o',
            color='green', markersize=8,
            markeredgecolor='black', markerfacecolor='green'
        )
    ax.set_xlim(0, room.LL[1])
    ax.set_ylim(0, room.LL[0])

    ax.imshow(likelihood_map, extent=extent, interpolation='bilinear')

    ax.set_title(title)
    ax.set_xlabel('Y-Coordinate')
    ax.set_ylabel('X-Coordinate')

    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Optionally, set the visibility of spines and grid.
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

