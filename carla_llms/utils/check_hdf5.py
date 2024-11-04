import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# put the path to the hdf5 file here
file_path = './data/Town01/train/episode_4.hdf5'

with h5py.File(file_path, 'r') as hdf_file:
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}")

            # Display 5 samples from the dataset if it's visualizable
            if obj.ndim >= 2 and obj.shape[0] >= 5:
                samples = obj[:5]
                if obj.ndim == 3 or (obj.ndim == 4 and obj.shape[-1] == 3):
                    for i, sample in enumerate(samples):
                        plt.figure()
                        plt.title(f'Sample {i+1} from {name}')
                        # if obj.ndim == 3:  # If grayscale or single-channel image
                        #     plt.imshow(sample, cmap='gray')
                        # else:  # For RGB images
                        #     plt.imshow(sample)
                        # plt.show()
                else:
                    print(f"First 5 samples from {name}: {samples}")
        elif isinstance(obj, h5py.Group):
            print(f"Group: {name}")

    hdf_file.visititems(print_structure)

distance_to_next_wp = []
distance_to_stop_line = []
distance_traveled = []
hlc = []
controls = []

with h5py.File(file_path, 'r') as hdf_file:
    distance_to_next_wp = hdf_file['distance_to_next_wp'][:]
    distance_to_stop_line = hdf_file['distance_to_stop_line'][:]
    distance_traveled = hdf_file['distance_traveled'][:]
    hlc = hdf_file['hlc'][:]
    controls = hdf_file['controls'][:]

for item in controls:
    print(item)

df = pd.DataFrame({
    'distance_to_next_wp': distance_to_next_wp.flatten(),
    'distance_to_stop_line': distance_to_stop_line.flatten(),
    'distance_traveled': distance_traveled.flatten(),
    'hlc': hlc.flatten()
})

csv_file_path = './episode_1_distances.csv'
df.to_csv(csv_file_path, index=False)

print(f"Data has been saved to {csv_file_path}")
