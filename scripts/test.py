import numpy as np
from pathlib import Path
def main():
    # Create a sample numpy array
    # data = np.load('data/Pointcept_data/flya_ship_place_v1_5/train/E3RDAA23480140_00197_052_1flight/coord.npy')
    # print("Data shape:", data.shape)
    coords = np.array([[0.01, 0.02, 0.03], [0.01, 0.03, 0.04], [0.02, 0.03, 0.01],
                       [0.04, 0.07, 0.02], [0.034, 0.08, 0.012],
                       [0.08, 0.015, -0.03]])
    coords = np.concatenate([coords, coords + 0.1, coords - 0.1, coords + 0.2, coords - 0.2, coords - 0.3], axis=0)
    vertex_labels = np.ones_like(coords[:, 0])

    colors = np.zeros_like(coords)

    normals = np.ones_like(coords)

    data_dict = dict(
        coord=coords.astype('float16'),
        color=colors.astype('uint8'),
        normal=normals.astype('float16'),
        segment=vertex_labels.astype('int8'),
    )

    # Save processed data
    output_folder = Path('data/Pointcept_data/test_data/train/scene_1')
    for key in data_dict.keys():
        np.save(output_folder / f'{key}.npy', data_dict[key])

if __name__ == "__main__":
    main()