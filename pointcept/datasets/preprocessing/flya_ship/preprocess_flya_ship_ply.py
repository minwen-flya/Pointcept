from typing import Iterable, Tuple
import open3d as o3d
import numpy as np
from pathlib import Path
import argparse
import os
import glob
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import pandas as pd
from collections import defaultdict


def discover_ply_paths(paths: Iterable[str]) -> list[str]:
    all_files = []
    for target_path in paths:
        p = Path(target_path)
        if p.is_file():
            all_files.append(str(p.resolve()))
        elif p.is_dir():
            all_files.extend(
                str(x.resolve())
                for x in p.rglob('*.ply')
                if x.name.lower().endswith('_livemap.ply')
            )
        else:
            raise FileNotFoundError(f'Path {target_path} does not exist.')

    if not all_files:
        raise FileNotFoundError('No matching PLY files found.')

    return sorted(all_files)


def downsample_point_cloud(
    coord: np.ndarray,
    normals: np.ndarray | None,
    xyz_orig: np.ndarray,
    voxel_size: float,
    color: np.ndarray | None,
    intensity: np.ndarray | None,
    label: np.ndarray | None,
) -> Tuple[
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """
    Voxel-based downsampling for point clouds.

    - Positions, colors, and normals are averaged within each voxel (if normals is not None).
    - Normals are re-normalized after averaging.
    - Discrete labels are aggregated using majority vote within each voxel.
    - Intensity is backfilled from the original cloud using nearest neighbor.

    Args:
        coord:     (N, 3) point positions.
        normals:   (N, 3) per-point normals, or None.
        color:     (N, 3) RGB in [0, 1] or None.
        intensity: (N,)   intensity or None.
        label:     (N,)   integer labels or None.
        xyz_orig:  (N, 3) original coordinates used to backfill intensity via NN.
        voxel_size: positive voxel size (meters).

    Returns:
        coord_ds:     (V, 3) downsampled positions.
        normals_ds:   (V, 3) downsampled, re-normalized normals or None.
        color_ds:     (V, 3) downsampled colors or None.
        intensity_ds: (V,)   intensity backfilled via NN or None.
        label_ds:     (V,)   majority-vote labels or None.
    """
    assert coord.shape[1] == 3
    N = coord.shape[0]

    if normals is not None:
        assert normals.shape[1] == 3 and normals.shape[0] == N

    if color is not None:
        assert color.shape == (N, 3)

    if intensity is not None:
        intensity = np.asarray(intensity)
        assert intensity.shape[0] == N

    if label is not None:
        label = np.asarray(label)
        assert label.shape[0] == N

    print(f'Applying spatial downsampling with voxel size: {voxel_size} m')

    # 1) Map points to voxel keys (integer grid)
    min_bound = coord.min(axis=0)
    voxel_idx = np.floor((coord - min_bound) / voxel_size).astype(np.int64)

    # Pack 3D voxel indices into a single 1D key for grouping
    keys_1d = voxel_idx[:, 0] * (10**12) + voxel_idx[:, 1] * (10**6) + voxel_idx[:, 2]

    # 2) Group by voxel
    uniq_keys, inverse, counts = np.unique(
        keys_1d, return_inverse=True, return_counts=True
    )
    V = uniq_keys.shape[0]

    # 3) Sum continuous attributes for averaging
    sum_pos = np.zeros((V, 3), dtype=np.float64)
    np.add.at(sum_pos, inverse, coord)
    coord_ds = (sum_pos / counts[:, None]).astype(np.float32)

    # Normals (optional)
    normals_ds = None
    if normals is not None:
        sum_nrm = np.zeros((V, 3), dtype=np.float64)
        np.add.at(sum_nrm, inverse, normals)
        normals_avg = sum_nrm / counts[:, None]

        # Re-normalize normals
        nrm_norm = np.linalg.norm(normals_avg, axis=1, keepdims=True)
        nrm_norm[nrm_norm == 0] = 1.0
        normals_ds = (normals_avg / nrm_norm).astype(np.float32)

    # Colors (optional)
    color_ds = None
    if color is not None:
        sum_col = np.zeros((V, 3), dtype=np.float64)
        np.add.at(sum_col, inverse, color)
        color_ds = (sum_col / counts[:, None]).astype(np.float32)

    # Labels (optional) — majority vote per voxel
    label_ds = None
    if label is not None:
        label_ds = np.empty(V, dtype=np.int32)
        # Build per-voxel index lists
        buckets = defaultdict(list)
        for i, v in enumerate(inverse):
            buckets[v].append(i)

        # Majority vote
        for v in range(V):
            lbls = np.asarray(label[buckets[v]]).ravel()
            if lbls.size == 0:
                label_ds[v] = -1
                continue

            # Fast path for non-negative smallish integers
            if lbls.min() >= 0 and int(lbls.max()) < 1e7:
                counts_lbl = np.bincount(lbls.astype(np.int64))
                label_ds[v] = np.argmax(counts_lbl)
            else:
                vals, cnts = np.unique(lbls, return_counts=True)
                label_ds[v] = vals[np.argmax(cnts)]

    # Intensity (optional) — nearest neighbor from original cloud
    intensity_ds = None
    if intensity is not None:
        pcd_orig = o3d.geometry.PointCloud()
        pcd_orig.points = o3d.utility.Vector3dVector(xyz_orig.astype(np.float64))
        kdtree = o3d.geometry.KDTreeFlann(pcd_orig)

        idx_nn = np.empty(V, dtype=np.int32)
        for i, p in enumerate(coord_ds):
            k, idx, _ = kdtree.search_knn_vector_3d(p.astype(np.float64), 1)
            idx_nn[i] = idx[0] if k > 0 else 0
        intensity_ds = intensity[idx_nn].astype(intensity.dtype, copy=False)

    print(f'Downsampled to {coord_ds.shape[0]} points')
    return coord_ds, normals_ds, color_ds, intensity_ds, label_ds


def estimate_normals(coord: np.ndarray) -> np.ndarray:
    """
    Estimate normals for a point cloud using Open3D.

    :param coord: Nx3 array of point coordinates
    :return: Nx3 array of normal vectors
    """
    print('Estimating normals on point cloud...')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)

    # Estimate normals with adaptive search parameters based on point cloud size
    num_points = coord.shape[0]
    if num_points > 100000:
        # For large point clouds, use smaller search parameters
        search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=20)
    else:
        # For smaller point clouds, use larger search parameters
        search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)

    pcd.estimate_normals(search_param=search_param)
    normals = np.asarray(pcd.normals).astype(np.float32)

    return normals


def handle_process(
    ply_path: str, output_path, mapping, train_scenes, val_scenes, voxel_size=None
):
    scene_id = Path(ply_path).parent.name
    data_name = scene_id

    output_path = Path(output_path)
    # Check which split the scene belongs to (train, val, or test)
    if scene_id in train_scenes:
        output_folder = output_path / 'train' / data_name
        split = 'train'
    elif scene_id in val_scenes:
        output_folder = output_path / 'val' / data_name
        split = 'val'
    else:
        output_folder = output_path / 'test' / data_name
        split = 'test'

    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    print(f'Processing: {data_name} in {split}')

    # Read PLY file
    pcd_ply = o3d.t.io.read_point_cloud(ply_path)
    # print(dir(pcd_ply.point))
    # Extract coordinates
    pc = pcd_ply.point

    coords = pc['positions'].numpy()
    try:
        vertex_labels = pc['label'].numpy()
    except KeyError:
        # skip this file if no label
        return None
    colors = np.zeros_like(coords, dtype=int)
    normals = pc['normals'].numpy()

    if voxel_size is not None:
        coords, normals, colors, _, vertex_labels = downsample_point_cloud(
            coord=coords,
            normals=normals,
            color=colors,
            intensity=None,
            xyz_orig=coords,
            voxel_size=voxel_size,
            label=vertex_labels,
        )

    if mapping is not None:
        vertex_labels = np.vectorize(mapping.get)(vertex_labels, vertex_labels)

    data_dict = dict(
        coord=coords.astype('float16'),
        color=colors.astype('uint8'),
        normal=normals.astype('float16'),
        segment=vertex_labels.astype('int8'),
    )

    # Save processed data
    for key in data_dict.keys():
        np.save(output_folder / f'{key}.npy', data_dict[key])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_root',
        type=str,
        nargs='+',
        required=True,
        help='Input PLY file(s) or directory(ies) containing PLY files.',
    )
    parser.add_argument(
        '--output_root',
        required=True,
        help='Output path where train/val folders will be located',
    )
    parser.add_argument(
        '--num_workers',
        default=mp.cpu_count(),
        type=int,
        help='Num workers for preprocessing.',
    )
    parser.add_argument(
        '--voxel_size',
        type=float,
        default=0.1,
        help='Num workers for preprocessing.',
    )
    opt = parser.parse_args()
    meta_root = Path(os.path.dirname(__file__)) / 'metadata'

    # Load label map
    category_mapping = pd.read_csv(
        meta_root / 'room_label_map.txt',
        sep='\t',
        header=None,
    )

    # Load train/val splits
    with open(meta_root / 'scenes_train.txt') as train_file:
        train_scenes = train_file.read().splitlines()
    with open(meta_root / 'scenes_val.txt') as val_file:
        val_scenes = val_file.read().splitlines()
    with open(meta_root / 'scenes_test.txt') as test_file:
        test_scenes = test_file.read().splitlines()

    # if opt.voxel_size:
    #     opt.output_root = opt.output_root + str(opt.voxel_size)
    # Create output directories
    os.makedirs(opt.output_root, exist_ok=True)
    train_output_dir = os.path.join(opt.output_root, 'train')
    os.makedirs(train_output_dir, exist_ok=True)
    val_output_dir = os.path.join(opt.output_root, 'val')
    os.makedirs(val_output_dir, exist_ok=True)
    test_output_dir = os.path.join(opt.output_root, 'test')
    os.makedirs(test_output_dir, exist_ok=True)

    scene_paths = discover_ply_paths(opt.dataset_root)

    mapping = {0: 0, 1: 1, 2: 2, 3: 0}
    # Preprocess data.
    pool = ProcessPoolExecutor(max_workers=opt.num_workers)
    print('Processing scenes...')
    _ = list(
        pool.map(
            handle_process,
            scene_paths,
            repeat(opt.output_root),
            repeat(mapping),
            repeat(train_scenes),
            repeat(val_scenes),
            repeat(opt.voxel_size),
        )
    )

# python pointcept/datasets/preprocessing/flya_ship/preprocess_flya_ship_ply.py --dataset_root ~/
# Documents/ship_datasets/GDrive/ ~/Documents/ship_datasets/flya_cloud --output_root data/flya_ship/
