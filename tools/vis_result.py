import os
import numpy as np
import open3d as o3d


def labels_to_colors(predicted_labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert label IDs to RGB colors.

    Args:
        predicted_labels: (N,) integer labels, may contain -1.
        num_classes: total number of classes.

    Returns:
        colors: (N, 3) float32 array in [0, 1].
    """
    # Fixed colors for specific classes
    fixed_colors = np.array(
        [
            [1.0, 0.0, 0.0],  # 0: red / bulk_carrier_cargo_tank
            [0.0, 1.0, 0.0],  # 1: green / deck
            [0.0, 0.0, 1.0],  # 2: blue / tanker_ballast_tank
            [1.0, 1.0, 0.0],  # 3: yellow / tanker_cargo_tank
        ],
        dtype=np.float32,
    )

    # Build colormap for 0..num_classes-1
    color_map = np.zeros((num_classes, 3), dtype=np.float32)
    num_fixed = fixed_colors.shape[0]
    use_fixed = min(num_fixed, num_classes)
    color_map[:use_fixed] = fixed_colors[:use_fixed]

    # Random colors for remaining classes
    if num_classes > num_fixed:
        rng = np.random.default_rng(0)
        color_map[num_fixed:num_classes] = rng.uniform(
            0.0, 1.0, size=(num_classes - num_fixed, 3)
        )

    # Ensure labels are np.ndarray
    predicted_labels = np.asarray(predicted_labels)

    # Output colors
    colors = np.zeros((predicted_labels.shape[0], 3), dtype=np.float32)

    # Handle normal labels (>=0)
    valid_mask = predicted_labels >= 0
    clipped_labels = np.clip(predicted_labels[valid_mask], 0, num_classes - 1)
    colors[valid_mask] = color_map[clipped_labels]

    # Handle -1 → white
    invalid_mask = predicted_labels < 0
    colors[invalid_mask] = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    return colors


if __name__ == "__main__":
    data_dir = "data/flya_ship"
    test_dir = os.path.join(data_dir, "val")
    test_pcd = "E300SA23130257_00448_310_1flight"

    res_dir = os.path.join(
        "exp",
        "sonata",
        "semseg-sonata-v1m1-0-base-0a-flya-lin",
        "result",
    )
    all_val_files = os.listdir(test_dir)
    for file in all_val_files:
        pred_path = os.path.join(res_dir, file + "_pred.npy")
        coord_path = os.path.join(test_dir, file + "/coord.npy")

        print("Loading:", pred_path)
        predicted_labels = np.load(pred_path)

        print("Loading:", coord_path)
        original_coord = np.load(coord_path)

        assert predicted_labels.shape[0] == original_coord.shape[0], \
            f"Label count {predicted_labels.shape[0]} != point count {original_coord.shape[0]}"

        pred_colors = labels_to_colors(predicted_labels, num_classes=4)

        print("Visualizing results...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(original_coord.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(pred_colors.astype(np.float64))

        # o3d.visualization.draw_geometries([pcd])

        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name="Predicted Semantics")
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()
