
from pathlib import Path
from pointcept.datasets.preprocessing.flya_ship.preprocess_flya_ship_ply import discover_ply_paths


def find_unlabeled_scenes():
    ply_paths = discover_ply_paths(
        [Path("/media/autonomy/f09b1330-40e1-4785-a821-5c7a98cf8196/ship_datasets/GDrive"), 
        Path("/media/autonomy/f09b1330-40e1-4785-a821-5c7a98cf8196/ship_datasets/flya_cloud"),
        Path("/media/autonomy/f09b1330-40e1-4785-a821-5c7a98cf8196/ship_datasets/stefano"),]
    )
    meta_root = Path("pointcept/datasets/preprocessing/flya_ship/metadata")
    with open(meta_root / 'scenes_train_place.txt') as train_file:
        train_scenes = train_file.read().splitlines()
    with open(meta_root / 'scenes_val_place.txt') as val_file:
        val_scenes = val_file.read().splitlines()
    with open(meta_root / 'scenes_test_place.txt') as test_file:
        test_scenes = test_file.read().splitlines()
    unlabeled_scenes = []
    for ply_path in ply_paths:
        name = ply_path.split('/')[-1].split('_livemap.ply')[0]
        if name not in train_scenes + val_scenes + test_scenes:
            unlabeled_scenes.append(name)
    print(f"Found {len(unlabeled_scenes)} unlabeled scenes:")
    with open(meta_root / "unlabeled_scenes.txt", "w") as out_file:
        for scene in unlabeled_scenes:
            out_file.write(f"{scene}\n")


if __name__ == '__main__':
    find_unlabeled_scenes()