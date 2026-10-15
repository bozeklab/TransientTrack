from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from traccuracy._tracking_graph import TrackingGraph
from traccuracy.matchers._base import Matcher


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of boxes.

    Boxes are ``[min_dim0, min_dim1, ..., max_dim0, max_dim1, ...]`` with
    exclusive maxima -- the layout traccuracy's ``bbox`` node attribute uses.
    Works for 2D (4 values) and 3D (6 values).
    """
    ndim = a.shape[1] // 2
    a_min, a_max = a[:, None, :ndim], a[:, None, ndim:]
    b_min, b_max = b[None, :, :ndim], b[None, :, ndim:]

    overlap = np.clip(np.minimum(a_max, b_max) - np.maximum(a_min, b_min), 0, None)
    inter = overlap.prod(axis=-1)

    a_area = (a[:, ndim:] - a[:, :ndim]).prod(axis=-1)[:, None]
    b_area = (b[:, ndim:] - b[:, :ndim]).prod(axis=-1)[None, :]
    union = a_area + b_area - inter

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


class BoxIOUMatcher(Matcher):
    """One-to-one matching on bounding-box IoU, solved per frame.

    Args:
        iou_threshold: minimum box IoU for a pair to be matchable.
        record_diagnostics: if True, store per-frame matching statistics on
            ``self.diagnostics`` for reporting.
    """

    _matching_type = "one-to-one"

    def __init__(self, iou_threshold: float = 0.5, record_diagnostics: bool = True):
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1]")
        self.iou_threshold = iou_threshold
        self.record_diagnostics = record_diagnostics
        self.diagnostics: dict[str, Any] = {}

    def _boxes(self, graph, nodes):
        try:
            return np.asarray(
                [graph.graph.nodes[n]["bbox"] for n in nodes], dtype=float
            )
        except KeyError:
            return self._boxes_from_segmentation(graph, nodes)

    def _boxes_from_segmentation(self, graph, nodes):
        """Derive bboxes from the label images (traccuracy < 0.4.0)."""
        if getattr(graph, "segmentation", None) is None:
            raise ValueError(
                "Graph has neither 'bbox' node attributes nor a segmentation array. "
                "Load with load_ctc_data(...) so the label images are attached."
            )
        if not hasattr(self, "_bbox_cache"):
            self._bbox_cache = {}
        key = id(graph)
        if key not in self._bbox_cache:
            cache = {}
            for t in range(graph.segmentation.shape[0]):
                frame = graph.segmentation[t]
                slices = ndimage.find_objects(frame)
                for label, sl in enumerate(slices, start=1):
                    if sl is not None:
                        cache[(t, label)] = [
                            sl[0].start,
                            sl[1].start,
                            sl[0].stop,
                            sl[1].stop,
                        ]
            self._bbox_cache[key] = cache

        cache = self._bbox_cache[key]
        out = []
        for n in nodes:
            attrs = graph.graph.nodes[n]
            out.append(cache[(attrs[graph.frame_key], attrs[graph.label_key])])
        return np.asarray(out, dtype=float)

    def _compute_mapping(self, gt_graph: TrackingGraph, pred_graph: TrackingGraph):
        mapping: list[tuple[Any, Any]] = []
        best_iou_per_gt: list[float] = []
        n_gt = n_pred = 0

        frames = sorted(set(gt_graph.nodes_by_frame) | set(pred_graph.nodes_by_frame))
        for t in frames:
            gt_nodes = list(gt_graph.nodes_by_frame.get(t, []))
            pred_nodes = list(pred_graph.nodes_by_frame.get(t, []))
            n_gt += len(gt_nodes)
            n_pred += len(pred_nodes)
            if not gt_nodes or not pred_nodes:
                best_iou_per_gt.extend([0.0] * len(gt_nodes))
                continue

            iou = box_iou(
                self._boxes(gt_graph, gt_nodes), self._boxes(pred_graph, pred_nodes)
            )
            best_iou_per_gt.extend(iou.max(axis=1).tolist())

            # Global one-to-one assignment, then drop pairs below threshold.
            # Maximising total IoU avoids the order-dependence of greedy matching.
            valid = iou >= self.iou_threshold
            if not valid.any():
                continue
            cost = np.where(valid, -iou, 0.0)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if valid[r, c]:
                    mapping.append((gt_nodes[r], pred_nodes[c]))

        if self.record_diagnostics:
            best = np.asarray(best_iou_per_gt) if best_iou_per_gt else np.zeros(0)
            self.diagnostics = {
                "iou_threshold": self.iou_threshold,
                "total_gt_nodes": n_gt,
                "total_pred_nodes": n_pred,
                "matched_nodes": len(mapping),
                "unmatched_gt_nodes": n_gt - len(mapping),
                "unmatched_pred_nodes": n_pred - len(mapping),
                "gt_node_recall": len(mapping) / n_gt if n_gt else float("nan"),
                "best_iou_percentiles": (
                    {
                        f"p{p}": float(np.percentile(best, p))
                        for p in (5, 25, 50, 75, 95)
                    }
                    if best.size
                    else {}
                ),
                "best_iou_mean": float(best.mean()) if best.size else float("nan"),
            }
        return mapping

    @property
    def info(self) -> dict[str, Any]:
        return {
            "name": "BoxIOUMatcher",
            "iou_threshold": self.iou_threshold,
            "matching type": self._matching_type,
        }
