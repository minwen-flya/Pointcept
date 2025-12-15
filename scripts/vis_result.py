import os
import argparse
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from pointcept.utils.misc import intersection_and_union


# =========================
# Config / Constants
# =========================

@dataclass
class ViewerConfig:
    dataset_root: str
    experiment_folder: Optional[str]
    experiment_name: Optional[str]
    ground_truth: bool
    ignore_outlier: bool
    view_iou: bool

    num_classes: int = 21
    unknown_index: int = -1
    # 可视化时隐藏的类 index（例如 outlier）
    invisible_index: List[int] = None
    # 计算 mean IoU 时忽略的类 index（例如 outlier + deck）
    iou_neglect_index: List[int] = None

    def __post_init__(self):
        if self.invisible_index is None:
            self.invisible_index = [0] if self.ignore_outlier else []
        if self.iou_neglect_index is None:
            self.iou_neglect_index = [0, 3]  # outlier and deck (按你原始逻辑)


DEFAULT_ID2CLASS = [
    "outlier",
    "pipe",
    "ladder",
    "deck",
    "bracket",
    "side shell plating",
    "inner seperate plating",
    "inner side plating",
    "inner bottom plating",
    "longitudinal web",
    "transverse web",
    "deck plating",
    "hopper plating",
    "bottom shell plating",
    "bulkheads",
    "horizontal girder",
    "longitudinals",
    "frames",
    "transverse stiffners",
    "inner ceiling plating",
]


# =========================
# Color mapping
# =========================

def load_color_map(csv_path: str) -> np.ndarray:
    """
    Load color map CSV with columns [r, g, b] in 0..255 (or 0..256-ish).
    Return float32 in [0, 1], shape (K, 3).
    """
    df = pd.read_csv(csv_path, header=0)
    color_map = np.array(df[["r", "g", "b"]], dtype=np.float32) / 256.0
    return color_map


def labels_to_colors(
    labels: np.ndarray,
    color_map: np.ndarray,
    invalid_color=(1.0, 1.0, 1.0),
):
    labels = labels.reshape(-1)
    colors = np.zeros((labels.shape[0], 3), dtype=np.float32)

    valid = labels >= 0
    labels_clip = np.clip(labels[valid], 0, color_map.shape[0] - 1)
    colors[valid] = color_map[labels_clip]
    colors[~valid] = invalid_color
    return colors

def build_local_to_global_map(
    num_global_classes: int,
    removed_global_ids: Sequence[int],
) -> np.ndarray:
    """
    Returns:
        local2global: shape (num_local_classes,)
    """
    removed = set(removed_global_ids)
    local2global = []

    for gid in range(num_global_classes):
        if gid not in removed:
            local2global.append(gid)

    return np.array(local2global, dtype=np.int64)


def remap_local_to_global(
    pred_local: np.ndarray,
    local2global: np.ndarray,
    unknown_index: int = -1,
) -> np.ndarray:
    pred_global = pred_local.copy()
    valid = pred_local >= 0
    pred_global[valid] = local2global[pred_local[valid]]
    pred_global[~valid] = unknown_index
    return pred_global

# =========================
# Dataset IO
# =========================

def list_val_cases(val_dir: str, max_cases: int = 10) -> List[str]:
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"val directory not found: {val_dir}")
    cases = os.listdir(val_dir)
    cases.sort()
    if len(cases) == 0:
        return []
    if len(cases) > max_cases:
        idx = np.random.choice(len(cases), size=max_cases, replace=False)
        cases = [cases[i] for i in idx]
    return cases


def build_pred_path(cfg: ViewerConfig, val_dir: str, case: str) -> str:
    if cfg.ground_truth:
        return os.path.join(val_dir, case, "segment.npy")
    res_dir = os.path.join("exp", cfg.experiment_folder, cfg.experiment_name, "result")
    return os.path.join(res_dir, case + "_pred.npy")


def load_case_arrays(cfg: ViewerConfig, val_dir: str, case: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_path = build_pred_path(cfg, val_dir, case)
    coord_path = os.path.join(val_dir, case, "coord.npy")
    gt_path = os.path.join(val_dir, case, "segment.npy")

    print("Loading:", pred_path)
    try:
        predicted = np.load(pred_path)
    except Exception as e:
        return np.array([]), np.array([]), np.array([])  # skip this case
    print("Predicted labels shape:", predicted.shape)

    print("Loading:", coord_path)
    coord = np.load(coord_path)
    print("Original coord shape:", coord.shape)

    if predicted.shape[0] != coord.shape[0]:
        raise ValueError(f"Label count {predicted.shape[0]} != point count {coord.shape[0]}")

    try:
        target = np.load(gt_path)
    except Exception as e:
        print(f"Error loading GT from {gt_path}: {e}")
        target = np.zeros_like(predicted) - 1  # all invalid
    return predicted, coord, target


# =========================
# Label / class utilities
# =========================
def filter_invisible(
    coord: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    invisible_index: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove points whose predicted label is in invisible_index.
    Return (coord_f, pred_f, target_f, keep_mask).
    """
    if len(invisible_index) == 0:
        keep = np.ones(pred.shape[0], dtype=bool)
        return coord, pred, target, keep

    keep = ~np.isin(pred, np.array(invisible_index, dtype=pred.dtype))
    return coord[keep], pred[keep], target[keep], keep


def print_per_class_iou(
    iou: np.ndarray,
    area_intersection: np.ndarray,
    area_union: np.ndarray,
    area_target: np.ndarray,
    id2class: List[str],
    invisible_index: Sequence[int],
    num_classes: int,
):
    for i in range(num_classes - 1):
        if i in invisible_index:
            continue

        shift = sum(1 for idx in invisible_index if idx < i)
        cls_idx = i - shift
        cls_name = id2class[cls_idx] if 0 <= cls_idx < len(id2class) else "Unknown"

        print(
            f"Class {i:2d} [{cls_name}]: IoU={iou[i]:.4f}, "
            f"Area Intersection={area_intersection[i]}, Area Union={area_union[i]}, Area Target={area_target[i]}"
        )

# =========================
# Open3D GUI app
# =========================

class PickerApp:
    def __init__(
        self,
        pcd: o3d.geometry.PointCloud,
        predicted_labels: np.ndarray,
        id2class: List[str],
        target_labels: np.ndarray,
    ) -> None:
        self.pcd = pcd
        self.predicted_labels = np.asarray(predicted_labels).reshape(-1)
        self.id2class = id2class
        self.target_labels = np.asarray(target_labels).reshape(-1)
        self.points_np = np.asarray(self.pcd.points)

        app = gui.Application.instance
        self.window = app.create_window("Predicted Semantics (Realtime)", 1024, 768)
        self.window.set_on_layout(self._on_layout)

        self.scene_widget = gui.SceneWidget()
        self.window.add_child(self.scene_widget)

        self.info_label = gui.Label("Shift + Left Click a point…")
        self.info_label.visible = True
        self.window.add_child(self.info_label)

        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)

        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 3 * self.window.scaling
        self.scene_widget.scene.add_geometry("pcd", self.pcd, mat)

        bounds = self.scene_widget.scene.bounding_box
        center = bounds.get_center()
        self.scene_widget.setup_camera(60.0, bounds, center)
        self.scene_widget.look_at(center, center - [0, 0, 3.0], [0, -1, 0])

        self.scene_widget.set_on_mouse(self._on_mouse_event)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self.scene_widget.frame = r

        pref = self.info_label.calc_preferred_size(layout_context, gui.Widget.Constraints())
        self.info_label.frame = gui.Rect(
            r.x,
            r.get_bottom() - pref.height,
            pref.width,
            pref.height,
        )

    def _label_to_name(self, label_id: int) -> str:
        if 0 <= label_id < len(self.id2class):
            return self.id2class[label_id]
        return "Unknown"

    def _on_mouse_event(self, event: gui.MouseEvent):
        # Shift + Left Click to pick point
        if (
            event.type == gui.MouseEvent.Type.BUTTON_DOWN
            and (event.buttons & int(gui.MouseButton.LEFT)) != 0
            and event.is_modifier_down(gui.KeyModifier.SHIFT)
        ):
            def depth_callback(depth_image: o3d.geometry.Image):
                x = event.x - self.scene_widget.frame.x
                y = event.y - self.scene_widget.frame.y
                depth = np.asarray(depth_image)[int(y), int(x)]

                if depth == 1.0:
                    text = "Clicked on background"
                else:
                    world = self.scene_widget.scene.camera.unproject(
                        float(x), float(y), float(depth),
                        self.scene_widget.frame.width,
                        self.scene_widget.frame.height,
                    )
                    dists = np.linalg.norm(self.points_np - world, axis=1)
                    idx = int(np.argmin(dists))

                    pred_id = int(self.predicted_labels[idx])
                    tgt_id = int(self.target_labels[idx])

                    text = (
                        f"idx={idx}, "
                        f"pred_id={pred_id}, pred_class={self._label_to_name(pred_id)}, "
                        f"target_id={tgt_id}, target_class={self._label_to_name(tgt_id)}"
                    )

                def update_label():
                    self.info_label.text = text
                    self.info_label.visible = True
                    print(text)

                gui.Application.instance.post_to_main_thread(self.window, update_label)

            self.scene_widget.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED

    def run(self):
        gui.Application.instance.run()


# =========================
# Main pipeline
# =========================

def visualize_case(
    cfg: ViewerConfig,
    case: str,
    val_dir: str,
    id2class: List[str],
    color_csv: str = "pointcept/datasets/preprocessing/flya_ship/metadata/custom20colors.csv",
):
    pred_local, coord, target = load_case_arrays(cfg, val_dir, case)
    if cfg.experiment_name and "ignore" in cfg.experiment_name:
        removed_global_ids = [0, 3]
        local2global = build_local_to_global_map(
            num_global_classes=cfg.num_classes,
            removed_global_ids=removed_global_ids,
        )
        pred = remap_local_to_global(pred_local, local2global, cfg.unknown_index)
    else:
        pred = pred_local
    if pred.shape[0] == 0:
        print(f"Skipping case {case} due to loading error.")
        return
    coord, pred, target, _keep = filter_invisible(coord, pred, target, cfg.invisible_index)

    print("labels after ignore:", np.unique(pred))

    # IoU
    area_intersection, area_union, area_target = intersection_and_union(
        pred, target, cfg.num_classes, ignore_index=(cfg.unknown_index,)
    )
    iou = area_intersection / (area_union + 1e-10)
    print_per_class_iou(iou, area_intersection, area_union, area_target, id2class, cfg.invisible_index, cfg.num_classes)
    valid = np.ones_like(iou, dtype=bool)
    valid[list(cfg.iou_neglect_index)] = False
    valid[area_union == 0] = False
    vals = iou[valid]
    print("Mean IoU:", float(np.mean(vals)))

    if not cfg.view_iou:
        color_map = load_color_map(color_csv)
        pred_colors = labels_to_colors(pred, color_map)
    else:
        # view_iou: pred==GT in red，others in black；GT ignored points also in black
        output = pred.reshape(-1).copy()
        tgt = target.reshape(-1)
        output[tgt == cfg.unknown_index] = cfg.unknown_index

        pred_colors = np.zeros((pred.shape[0], 3), dtype=np.float32)
        match = (output == tgt) & (tgt != cfg.unknown_index)
        pred_colors[match] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(pred_colors.astype(np.float64))

    print("Visualizing results...")
    gui.Application.instance.initialize()
    PickerApp(pcd, pred, id2class, target).run()


def parse_args() -> ViewerConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_root", type=str, required=True)
    parser.add_argument("-f", "--experiment_folder", type=str)
    parser.add_argument("-n", "--experiment_name", type=str)
    parser.add_argument("-g", "--ground_truth", action="store_true", help="Enable ground truth visualization.")
    parser.add_argument("--ignore_outlier", action="store_true")
    parser.add_argument("--view_iou", action="store_true")
    opt = parser.parse_args()

    return ViewerConfig(
        dataset_root=opt.dataset_root,
        experiment_folder=opt.experiment_folder,
        experiment_name=opt.experiment_name,
        ground_truth=opt.ground_truth,
        ignore_outlier=opt.ignore_outlier,
        view_iou=opt.view_iou,
    )


def main():
    cfg = parse_args()
    val_dir = os.path.join(cfg.dataset_root, "val")

    cases = list_val_cases(val_dir, max_cases=10)
    if len(cases) == 0:
        print("No files found in", val_dir)
        return
    print(f"Found {len(cases)} cases to visualize.")

    for case in cases:
        visualize_case(cfg, case, val_dir, id2class=DEFAULT_ID2CLASS.copy())


if __name__ == "__main__":
    main()
