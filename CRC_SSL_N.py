import argparse
from Code.crc_ssl import CoverageSet
from Code.plots import plot_roi_neighbours
from Code.utilities import *

parser = argparse.ArgumentParser(description="Run CRC_SSL_N script with configurable parameters.")
parser.add_argument("--plot", type=int, default=0, help="Enable or disable plotting (0 or 1)")
parser.add_argument("--model_type", type=str, default="SRP_DNN", help="Model type (SRP_PHAT or SRP_DNN)")
parser.add_argument("--num_iterations", type=int, default=100, help="Number of iterations")
parser.add_argument("--seed", type=int, default=1234567890, help="Random seed")
parser.add_argument("--snr", type=int, default=15, help="Signal-to-noise ratio")
parser.add_argument("--reverb", type=int, default=700, help="Reverberation time in ms")
parser.add_argument("--speakers", type=int, default=3, help="Number of speakers")
parser.add_argument("--Kmax", type=int, default=3, help="Maximum speakers count")
parser.add_argument("--lambda_steps", type=int, default=1000, help="Number of steps for lambda")
parser.add_argument("--significance_levels", type=float, nargs='+', default=[0.1, 0.05], help="Significance levels")
args = parser.parse_args()

plot = args.plot
model_type = args.model_type
num_iterations = args.num_iterations
seed = args.seed
np.random.seed(seed)

snr = args.snr
reverb = args.reverb
speakers = args.speakers
Kmax = args.Kmax

lambda_list_ext = np.linspace(0., 1., args.lambda_steps) # Lambda
significance_level = np.array(args.significance_levels)

filename = f'./data/{model_type}/Synthetic/Reverb_{reverb}_ms_SNR_{snr}_dB/speakers_{speakers}.npz'

print(f'Current setup: {model_type} with {speakers} speakers')

data = np.load(filename, allow_pickle=True)
speaker_pos = data['speaker_pos']
all_estimated_positions = data['all_estimated_positions']
all_likelihood_maps = data['all_likelihood_maps']

room_obj = data['rir_obj'].item()
room = type('Room', (object,), room_obj)()
grid_size = room.xl.size
total_dataset_size = speaker_pos.shape[0]

splits = generate_random_splits(total_samples=total_dataset_size,
                                num_iterations=num_iterations,
                                calib_size=int(total_dataset_size * 0.8),
                                num_lists=1, random_seed=42)
folds_across_lists = list(zip(*splits))

coverage_array, area_array = [], []

for iter in range(num_iterations):
    print(f"Fold {iter}:")
    calib_index, test_index = folds_across_lists[iter][0]

    if plot:
        dest_path = create_save_directory(f'{model_type}_fold_{iter}')
    else:
        dest_path = None
    cov_set_obj = CoverageSet(true_position=speaker_pos[calib_index, ...],
                              estimated_positions=all_estimated_positions[calib_index, :speakers, ...],
                              likelihood_maps=all_likelihood_maps[calib_index, :speakers, ...],
                              lambda_list=lambda_list_ext, room=room, path_=dest_path, plot_function=plot_roi_neighbours)
    cov_set_obj.calibrate(plot=False, plot_coverage_set=0)


    #[Folds, Speakers, Significance Levels]
    coverage, area = cov_set_obj.test(test_sets=test_index.size,
                                      true_positions=speaker_pos[test_index, ...],
                                      estimated_positions=all_estimated_positions[test_index, :speakers, ...],
                                      likelihood_maps=all_likelihood_maps[test_index, :speakers, ...],
                                      significance_level=significance_level,
                                      test_plot=False)

    coverage_array.append(coverage)
    area_array.append(area/grid_size*100)
    print(coverage)
    print(area)

print('Final Results')
coverage_array = np.mean(coverage_array, axis=0)
area_array = np.mean(area_array, axis=0)
print(coverage_array)
print(area_array)

print_results(coverage_array=coverage_array,
              area_array=area_array,
              significance_level=significance_level,
              area_unit='% of grid area')
