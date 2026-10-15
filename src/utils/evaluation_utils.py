from __future__ import annotations

import csv
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import traccuracy
from ctc_metrics.metrics import lnk
from ctc_metrics.scripts.evaluate import load_data
from ctc_metrics.utils.representations import count_acyclic_graph_correction_operations
from lightning.pytorch.loggers.logger import Logger
from scipy.optimize import linear_sum_assignment
from skimage.morphology import remove_small_objects
from traccuracy import run_metrics
from traccuracy.loaders import load_ctc_data
from traccuracy.metrics import DivisionMetrics
from trackeval import metrics

from utils import misc
from utils.box_matcher import BoxIOUMatcher
from utils.ctc.ctc_eval import ctc_score


def fully_occludes(new_box, existing_box):
    new_x1, new_y1, new_x2, new_y2 = new_box
    existing_x1, existing_y1, existing_x2, existing_y2 = existing_box

    # Check for full occlusion
    return (
        new_x1 <= existing_x1
        and new_x2 >= existing_x2
        and new_y1 <= existing_y1
        and new_y2 >= existing_y2
    )


def resolve_duplicate_boxes(bounding_boxes):
    flipped_boxes = {}
    for key, value in bounding_boxes.items():
        if tuple(value) in flipped_boxes:
            flipped_boxes[tuple(value)].append(key)
        else:
            flipped_boxes[tuple(value)] = [key]

    duplicate_boxes = [keys for keys in flipped_boxes.values() if len(keys) > 1]

    for duplicates in duplicate_boxes:
        for i, duplicate in enumerate(duplicates):
            print("Duplicate boxes found!")
            if i % 2 == 0:
                bounding_boxes[duplicate] = [
                    x + (len(duplicates) - 1) * 1 for x in bounding_boxes[duplicate]
                ]
            else:
                bounding_boxes[duplicate] = [
                    x - (len(duplicates) - 1) * 1 for x in bounding_boxes[duplicate]
                ]


def load_mask(data_dir, image_path):
    mask = cv2.imread(os.path.join(data_dir, image_path), -1).astype(np.float32)

    mask = remove_small_objects(mask > 0, min_size=60, connectivity=1)
    return mask


def combine_mask_labels(
    tracklets, start_time, end_time, ctc_origin, resize_ratio, output_dir
):
    """Combines the mask and label images into a single image.

    Parameters
    ----------
    tracklets : list
        List of tracklets.
    data_type : str
        Type of the image sequence.
    mask_dir : str
        Path to the folder containing the segmentation masks.
    output_dir : str
        Path to the folder where the combined images will be saved.
    """
    scaling_factor = 1.0

    for i in range(start_time, end_time + 1):
        if resize_ratio:
            img_size = [int(1024 / resize_ratio), int(1024 / resize_ratio)]
        else:
            img_size = [1024, 1024]
        transformed_mask = np.zeros(img_size, dtype=np.uint16)

        bounding_boxes = {}

        for track in tracklets:
            if track.t_start <= i <= track.t_end:
                cell = track.get_cell(i)

                # Compute the bounding box coordinates and clamp them within the mask dimensions
                x_min, y_min = max(
                    0, min(cell.bounding_box[0], cell.bounding_box[2])
                ), max(0, min(cell.bounding_box[1], cell.bounding_box[3]))
                x_max, y_max = min(
                    img_size[1], max(cell.bounding_box[0], cell.bounding_box[2])
                ), min(img_size[0], max(cell.bounding_box[1], cell.bounding_box[3]))

                # Calculate the center of the bounding box
                x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2

                # Apply the scaling factor to adjust the bounding box size around the center
                new_width, new_height = scaling_factor * (
                    x_max - x_min
                ), scaling_factor * (y_max - y_min)

                x_min, x_max = max(0, int(x_center - new_width / 2)), min(
                    img_size[1], int(x_center + new_width / 2)
                )
                y_min, y_max = max(0, int(y_center - new_height / 2)), min(
                    img_size[0], int(y_center + new_height / 2)
                )

                x_min = max(min(img_size[1] - 1, x_min), 0)
                x_max = min(max(1, x_max), img_size[1])
                y_min = max(min(img_size[0] - 1, y_min), 0)
                y_max = min(max(1, y_max), img_size[0])

                transformed_mask[y_min:y_max, x_min:x_max] = cell.id

                bounding_boxes[cell.id] = [x_min, y_min, x_max, y_max]

        # check for the rare case where bounding boxes have the same size
        resolve_duplicate_boxes(bounding_boxes)

        all_cells = list(bounding_boxes.keys())

        for cell_id, [x_min, y_min, x_max, y_max] in bounding_boxes.items():
            if cell_id not in transformed_mask:
                transformed_mask[y_min:y_max, x_min:x_max] = cell_id
                already_drawn = [x for x in all_cells if x <= cell_id and x != 0]
                for drawn_cell in already_drawn:
                    if drawn_cell not in transformed_mask:
                        bbox = bounding_boxes[drawn_cell]
                        transformed_mask[bbox[1] : bbox[3], bbox[0] : bbox[2]] = (
                            drawn_cell
                        )

        if resize_ratio:
            transformed_mask = cv2.resize(
                transformed_mask,
                (1024, 1024),
                interpolation=cv2.INTER_NEAREST,
            )

        cv2.imwrite(
            os.path.join(
                output_dir, f"mask{str(i-start_time + ctc_origin).zfill(3)}.tif"
            ),
            transformed_mask,
        )


def mot_to_hota_data(gt, tr, num_frames=None):
    if num_frames is None:
        num_frames = int(max(gt[:, 0].max(), tr[:, 0].max()))

    gt_ids, tracker_ids, similarity_scores = [], [], []

    all_gt_ids = sorted(set(gt[:, 1]) - {-1})  # remove -1 (ignore regions)
    all_tr_ids = sorted(set(tr[:, 1]))

    gt_id_map = {id_: idx for idx, id_ in enumerate(all_gt_ids)}
    tr_id_map = {id_: idx for idx, id_ in enumerate(all_tr_ids)}

    for f in range(1, num_frames + 1):
        # gt_f = gt[gt[:, 0] == f]
        gt_f = gt[(gt[:, 0] == f) & (gt[:, 1] != -1)]
        tr_f = tr[tr[:, 0] == f]

        # ids
        gt_ids_t = np.array([gt_id_map[i] for i in gt_f[:, 1] if i != -1], dtype=int)
        tr_ids_t = np.array([tr_id_map[i] for i in tr_f[:, 1]], dtype=int)

        gt_ids.append(gt_ids_t)
        tracker_ids.append(tr_ids_t)

        # boxes [x1,y1,x2,y2]
        gt_boxes = (
            np.stack(
                [
                    gt_f[:, 2],
                    gt_f[:, 3],
                    gt_f[:, 2] + gt_f[:, 4],
                    gt_f[:, 3] + gt_f[:, 5],
                ],
                axis=1,
            )
            if len(gt_f) > 0
            else np.zeros((0, 4))
        )
        tr_boxes = (
            np.stack(
                [
                    tr_f[:, 2],
                    tr_f[:, 3],
                    tr_f[:, 2] + tr_f[:, 4],
                    tr_f[:, 3] + tr_f[:, 5],
                ],
                axis=1,
            )
            if len(tr_f) > 0
            else np.zeros((0, 4))
        )

        # compute IoU matrix
        sim = np.zeros((len(gt_boxes), len(tr_boxes)))
        for i, g in enumerate(gt_boxes):
            gx1, gy1, gx2, gy2 = g
            ga = (gx2 - gx1) * (gy2 - gy1)
            for j, d in enumerate(tr_boxes):
                dx1, dy1, dx2, dy2 = d
                da = (dx2 - dx1) * (dy2 - dy1)

                inter_x1, inter_y1 = max(gx1, dx1), max(gy1, dy1)
                inter_x2, inter_y2 = min(gx2, dx2), min(gy2, dy2)
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                else:
                    inter = 0
                union = ga + da - inter
                sim[i, j] = inter / union if union > 0 else 0
        similarity_scores.append(sim)

    return {
        "num_timesteps": num_frames,
        "num_gt_dets": len(gt),
        "num_tracker_dets": len(tr),
        "num_gt_ids": len(all_gt_ids),
        "num_tracker_ids": len(all_tr_ids),
        "gt_ids": gt_ids,
        "tracker_ids": tracker_ids,
        "similarity_scores": similarity_scores,
    }


def trackeval_evaluation(
    tracklets, start_time, end_time, data_dir, eval_dir, position, resize_ratio
):
    # load gt tracks
    gt = np.loadtxt(
        os.path.join(data_dir, position, f"motm_{position}.csv"),
        delimiter=",",
        skiprows=1,
    )

    if gt is None:
        print("No ground truth available for MOT evaluation!")
        return

    if resize_ratio:
        gt[:, 2:5] /= resize_ratio

    # OFFSET = 48  # annotations start at frame 49
    # OFFSET = 0

    # load predictions in mot format
    preds = []

    for time_step in range(start_time - 1, end_time + 1):
        candidates = [
            tracklet.get_cell(time_step)
            for tracklet in tracklets
            if tracklet.get_cell(time_step) is not None
        ]

        for candidate in candidates:
            if candidate.alive:
                x1, y1, x2, y2 = candidate.bounding_box
                w, h = x2 - x1, y2 - y1

                preds.append(
                    [
                        time_step,
                        candidate.id,
                        x1,
                        y1,
                        w,
                        h,
                        1.0,
                        -1,
                        -1,
                        -1,
                    ]
                )

    preds = np.array(preds, dtype=int)
    np.savetxt(
        os.path.join(eval_dir, "mot_preds.csv"),
        preds,
        fmt="%.2f",
        delimiter=",",
        header="frame_number, object_id, bb_left, bb_top, bb_width, bb_height, confidence, x, y, z",
        comments="",
    )

    data = mot_to_hota_data(gt, preds)

    results = {}

    for metric in [
        metrics.HOTA(),
        metrics.CLEAR({"PRINT_CONFIG": False}),
        metrics.Identity({"PRINT_CONFIG": False}),
    ]:
        results[metric.get_name()] = metric.eval_sequence(data)

    output = {
        "HOTA": results["HOTA"]["HOTA"].mean(),
        "DetA": results["HOTA"]["DetA"].mean(),
        "AssA": results["HOTA"]["AssA"].mean(),
        "MOTA": results["CLEAR"]["MOTA"],
        "MOTP": results["CLEAR"]["MOTP"],
        "IDF1": results["Identity"]["IDF1"],
        "NUM_TRACKS": results["CLEAR"]["MT"]
        + results["CLEAR"]["PT"]
        + results["CLEAR"]["ML"],
    }

    print("TrackEval results:")
    for key, value in output.items():
        if key == "NUM_TRACKS":
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.2%}")

    return output, gt


def apoptosis_evaluation(
    coco, tracklets, video_id, start_time, end_time, delta, window=2
):
    gt_tracks = build_gt_tracks(coco, video_id)

    pred_tracks = {}

    for tracklet in tracklets:
        track_id = tracklet.tracklet_id
        pred_tracks[track_id] = {}

        for t in range(start_time - 1, end_time + 1):
            cell = tracklet.get_cell(t)
            if cell is None:
                continue

            x1, y1, x2, y2 = cell.bounding_box
            # pred_tracks[track_id][t] = (x1, y1, x2 - x1, y2 - y1)
            pred_tracks[track_id][t] = (x1, y1, x2, y2)

    matches = get_matches_hung(gt_tracks, pred_tracks, window=window)
    # matches = get_matches_greedy(gt_tracks, pred_tracks)
    # print(matches)
    # print("Number of matches: ", len(matches))

    tracklet_dict = {tracklet.tracklet_id: tracklet for tracklet in tracklets}

    # statistics
    gt_counter = 0
    gt_deaths = []
    pred_counter = 0
    pred_deaths = []

    for _, track in gt_tracks.items():
        if track["time_of_death"] is not None:
            gt_counter += 1
            gt_deaths.append(track["time_of_death"])
    for pred_id in pred_tracks.keys():
        if tracklet_dict[pred_id].time_of_death is not None:
            pred_counter += 1
            pred_deaths.append(tracklet_dict[pred_id].time_of_death)

    gt_deaths = sorted(gt_deaths)
    pred_deaths = sorted(pred_deaths)

    apoptosis_results = evaluate_event(
        matches, gt_tracks, tracklet_dict, delta=delta, event="apoptosis"
    )

    return apoptosis_results


def build_gt_tracks(coco, video_id):
    gt_tracks = defaultdict(
        lambda: {"frames": {}, "time_of_death": None, "time_of_division": None}
    )

    img_ids = coco.get_img_ids_from_vid(video_id)

    for img_id in img_ids:
        for ann in coco.imgToAnns[img_id]:

            cid = ann["cell_id"]
            t = ann["time_step"] + 1

            bbox = [x * 1024 for x in ann["bbox"]]
            cx, cy, w, h = bbox
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            gt_tracks[cid]["frames"][t] = (x1, y1, x2, y2)

            if ann.get("time_of_death") is not None:
                gt_tracks[cid]["time_of_death"] = ann["time_of_death"] + 1

            if ann.get("time_of_division") is not None:
                gt_tracks[cid]["time_of_division"] = ann["time_of_division"] + 1

    return gt_tracks


def evaluate_event(matches, gt_tracks, tracklets, delta=5, event="apoptosis"):
    TP, FP, FN, TN = 0, 0, 0, 0
    timing_errors = []

    matched_missed_timing, matched_missed_atall = 0, 0
    missed_at_all = []

    for gt_id, pred_id in matches.items():
        if event == "apoptosis":
            gt_event = gt_tracks[gt_id]["time_of_death"]
            pred_event = tracklets[pred_id].time_of_death
        elif event == "mitosis":
            gt_event = gt_tracks[gt_id]["time_of_division"]
            pred_event = tracklets[pred_id].time_of_division

        # 1. GT has event
        if gt_event is not None:
            if pred_event is not None:  # predicted also has event
                if abs(gt_event - pred_event) <= delta:
                    TP += 1
                    timing_errors.append(abs(gt_event - pred_event))
                else:
                    FP += 1
                    FN += 1
                    matched_missed_timing += 1
            else:  # predicted missing event
                FN += 1
                matched_missed_atall += 1
                missed_at_all.append(gt_id)
        else:  # GT has no event
            if pred_event is not None:  # hallucinated event
                FP += 1
            else:  # correctly predicts no event
                TN += 1

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    mean_timing_error = np.mean(timing_errors) if len(timing_errors) > 0 else None

    return {
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "mean_timing_error": mean_timing_error,
        "timing_errors": ",".join(str(e) for e in timing_errors),
        "n_timing_excluded": matched_missed_timing,
    }


def get_gt_event(track, _id, event):
    if event == "apoptosis":
        return track[_id]["time_of_death"]
    elif event == "mitosis":
        return track[_id]["time_of_division"]
    else:
        raise Exception("Unknown event")


def get_pred_event(track, _id, event):
    if event == "apoptosis":
        return track[_id].time_of_death
    elif event == "mitosis":
        return track[_id].time_of_division
    else:
        raise Exception("Unknown event")


def get_matches_greedy(gt_tracks, pred_tracks, min_iou=0.1):
    matches = {}
    used_preds = set()

    for gt_id, gt_track in gt_tracks.items():
        best_pred = None
        best_score = 0

        for pred_id, pred_track in pred_tracks.items():
            score = track_iou_spatio_temporal_window(
                gt_track, pred_track, gt_tracks[gt_id]["time_of_death"]
            )

            if pred_id in used_preds:
                continue

            if score > best_score:
                best_score = score
                best_pred = pred_id

            if best_pred is not None and best_score >= min_iou:
                matches[gt_id] = best_pred
                used_preds.add(best_pred)

    return matches


def get_matches_hung(gt_tracks, pred_tracks, min_iou=0.1, window=2):
    gt_ids = list(gt_tracks.keys())
    pred_ids = list(pred_tracks.keys())

    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return {}

    cost_matrix = np.ones((len(gt_ids), len(pred_ids)))

    for i, gt_id in enumerate(gt_ids):
        for j, pred_id in enumerate(pred_ids):

            iou = track_iou_spatio_temporal_window(
                gt_tracks[gt_id],
                pred_tracks[pred_id],
                gt_tracks[gt_id]["time_of_death"],
                window=window,
            )

            cost_matrix[i, j] = 1.0 - iou

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = {}

    for r, c in zip(row_ind, col_ind):
        iou = 1.0 - cost_matrix[r, c]

        if iou >= min_iou:
            gt_id = gt_ids[r]
            pred_id = pred_ids[c]

            matches[gt_id] = pred_id

    return matches


def get_track_time_overlap(gt_track, pred_track, start_time, end_time):
    gt_times = set(gt_track.keys())
    pred_times = set(pred_track.keys())
    overlap = gt_times.intersection(pred_times)
    return len(overlap)


def track_iou_overlap_mean(gt_track, pred_track, min_overlap_frames=2):
    times = set(gt_track["frames"].keys()).intersection(set(pred_track.keys()))
    if len(times) < min_overlap_frames:
        return 0.0

    ious = []
    for t in times:
        gx1, gy1, gx2, gy2 = gt_track["frames"][t]
        px1, py1, px2, py2 = pred_track[t]

        xx1 = max(gx1, px1)
        yy1 = max(gy1, py1)
        xx2 = min(gx2, px2)
        yy2 = min(gy2, py2)

        inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)

        g_area = max(0, gx2 - gx1) * max(0, gy2 - gy1)
        p_area = max(0, px2 - px1) * max(0, py2 - py1)

        union = g_area + p_area - inter

        if union <= 0:
            iou = 0.0
        else:
            iou = inter / union

        ious.append(iou)

    return np.mean(ious)


def track_iou_aggregate(gt_track, pred_track, min_overlap_frames=2):
    times = set(gt_track["frames"]).intersection(pred_track)

    if len(times) < min_overlap_frames:
        return 0.0

    total_inter = 0.0
    total_union = 0.0

    for t in times:
        gx1, gy1, gx2, gy2 = gt_track["frames"][t]
        px1, py1, px2, py2 = pred_track[t]

        xx1 = max(gx1, px1)
        yy1 = max(gy1, py1)
        xx2 = min(gx2, px2)
        yy2 = min(gy2, py2)

        inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)

        g_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
        p_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)

        union = g_area + p_area - inter

        total_inter += inter
        total_union += union

    return total_inter / total_union if total_union > 0 else 0.0


def track_iou_spatio_temporal_window(gt_track, pred_track, event_frame, window=2):
    gt_frames = gt_track["frames"]

    if event_frame != None:
        frames = range(event_frame - window, event_frame + window + 1)
    else:
        gt_times = set(gt_frames)
        pred_times = set(pred_track)

        frames = gt_times | pred_times

    total_inter = 0.0
    total_union = 0.0

    for t in frames:
        gt_box = gt_frames.get(t)
        pred_box = pred_track.get(t)

        if gt_box is None or pred_box is None:
            continue

        gx1, gy1, gx2, gy2 = gt_box
        px1, py1, px2, py2 = pred_box

        xx1 = max(gx1, px1)
        yy1 = max(gy1, py1)
        xx2 = min(gx2, px2)
        yy2 = min(gy2, py2)

        inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)

        g_area = (gx2 - gx1) * (gy2 - gy1)
        p_area = (px2 - px1) * (py2 - py1)

        union = g_area + p_area - inter

        total_inter += inter
        total_union += union

    return total_inter / total_union if total_union > 0 else 0.0


def track_iou_spatio_temporal(gt_track, pred_track, min_overlap_frames=2):
    gt_frames = gt_track["frames"]

    gt_times = set(gt_frames)
    pred_times = set(pred_track)

    union_times = gt_times | pred_times

    total_inter = 0.0
    total_union = 0.0

    for t in union_times:

        gt_box = gt_frames.get(t)
        pred_box = pred_track.get(t)

        if gt_box is None:
            px1, py1, px2, py2 = pred_box
            p_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)

            total_union += p_area
            continue

        if pred_box is None:
            gx1, gy1, gx2, gy2 = gt_box
            g_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)

            total_union += g_area
            continue

        gx1, gy1, gx2, gy2 = gt_box
        px1, py1, px2, py2 = pred_box

        xx1 = max(gx1, px1)
        yy1 = max(gy1, py1)
        xx2 = min(gx2, px2)
        yy2 = min(gy2, py2)

        inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)

        g_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
        p_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)

        union = g_area + p_area - inter

        total_inter += inter
        total_union += union

    return total_inter / total_union if total_union > 0 else 0.0


def ctc_evaluation(results_dir, gt_dir, subfolder, delete_tmp_files=False):
    """Evaluates the tracking results using the CTC metric.

    Parameters
    ----------
    results_dir : str
        Path to the folder containing the tracking results.
    gt_dir : str
        Path to the folder containing the ground truth results.
    delete_tmp_files : bool, optional
        Whether to delete the temporary files, by default False
    """
    evaluation_dir = Path(Path(results_dir).parent, "tmp_evaluation")

    ctc_results = ctc_score(
        Path(results_dir),
        Path(gt_dir),
        evaluation_dir,
        subfolder,
    )

    print("LNK score calculation started...")
    # shift gt and res files to start at 000. Accordingly, adjust the track file
    shift_files(evaluation_dir, subfolder)

    comp_tracks, ref_tracks, traj, _, _, _ = load_data(
        Path(Path(results_dir).parent, "tmp_evaluation", subfolder + "_RES/"),
        Path(Path(results_dir).parent, "tmp_evaluation", subfolder + "_GT/"),
        True,
        False,
    )

    graph_operations = count_acyclic_graph_correction_operations(
        ref_tracks,
        comp_tracks,
        traj["labels_ref"],
        traj["labels_comp"],
        traj["mapped_ref"],
        traj["mapped_comp"],
    )

    ctc_results["LNK"] = lnk(**graph_operations)
    print(f"LNK score: {ctc_results['LNK']:.6f}")

    print("Division metrics calculation started...")
    try:
        record = evaluate_dirs(
            Path(evaluation_dir, subfolder + "_GT"),
            Path(evaluation_dir, subfolder + "_RES"),
            name=subfolder,
        )
        ctc_results.update(flat_fields(record))
        print(format_record(record, buffer=0))
    except Exception as exc:
        print(f"Division metrics failed for {subfolder}: {exc}")

    if delete_tmp_files:  # delete temporary files:
        shutil.rmtree(evaluation_dir, onerror=misc.remove_readonly)
        shutil.rmtree(results_dir, onerror=misc.remove_readonly)

    return ctc_results


def shift_files(evaluation_dir, subfolder):
    gt_folder = Path(evaluation_dir, f"{subfolder}_GT", "TRA")
    res_folder = Path(evaluation_dir, f"{subfolder}_RES")
    gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith(".tif")])
    gt_track_file = Path(gt_folder, "man_track.txt")

    res_files = sorted([f for f in os.listdir(res_folder) if f.endswith(".tif")])
    res_track_file = Path(res_folder, "res_track.txt")

    gt_offset = int(gt_files[0].split("track")[1].split(".tif")[0])
    res_offset = int(res_files[0].split("mask")[1].split(".tif")[0])

    assert len(gt_files) == len(
        res_files
    ), f"GT has {len(gt_files)} frames, RES has {len(res_files)}"

    for file in gt_files:
        frame_number = int(file.split("track")[1].split(".tif")[0])
        new_frame_number = frame_number - gt_offset
        new_file_name = f"man_track{str(new_frame_number).zfill(3)}.tif"
        os.rename(
            Path(gt_folder, file),
            Path(gt_folder, new_file_name),
        )

    for file in res_files:
        frame_number = int(file.split("mask")[1].split(".tif")[0])
        new_frame_number = frame_number - res_offset
        new_file_name = f"mask{str(new_frame_number).zfill(3)}.tif"
        os.rename(
            Path(res_folder, file),
            Path(res_folder, new_file_name),
        )

    for trackfile, off in [(gt_track_file, gt_offset), (res_track_file, res_offset)]:
        data = np.loadtxt(trackfile)
        data[:, 1] -= off
        data[:, 2] -= off
        data = data.astype(int)
        np.savetxt(trackfile, data, fmt="%d %d %d %d")


def ctc_gt_origin(gt_dir):
    """First frame number used by the CTC ground truth (0, 1, 49, ...)."""
    d = Path(gt_dir)
    tra = d / "TRA" if (d / "TRA").is_dir() else d
    files = sorted(
        f for f in os.listdir(tra) if f.startswith("man_track") and f.endswith(".tif")
    )
    if not files:
        raise FileNotFoundError(f"No man_track*.tif in {tra}")
    return int(files[0][len("man_track") : -len(".tif")])


def write_ctc_files(tracklets, output_dir, start_id, ctc_origin, end_id):
    """Converts the summary file to the res_track.txt file required and used for ctc evaluation.

    Parameters
    ----------
    tracklets : list
        List of tracklets.
    output_dir : str
        Path to the folder where the converted summary file will be saved.
    """
    # delete old file
    ctc_output_file = os.path.join(output_dir, "ctc_files", "res_track.txt")
    if os.path.isfile(ctc_output_file):
        os.remove(ctc_output_file)

    with open(ctc_output_file, "a") as file:
        for tracklet in tracklets:
            end = tracklet.t_end if tracklet.t_end is not None else end_id
            parent = tracklet.parent if tracklet.parent is not None else 0
            shift = ctc_origin - start_id
            file.write(
                f"{tracklet.tracklet_id} {tracklet.t_start + shift} {end +shift} {parent}\n"
            )


def log_motmetrics(logger: Logger, results):
    columns = [
        "HOTA",
        "DetA",
        "AssA",
        "MOTA",
        "MOTP",
        "IDF1",
    ]

    formatter_dict = {
        "HOTA": "{:,.2%}".format,
        "DetA": "{:,.2%}".format,
        "AssA": "{:,.2%}".format,
        "MOTA": "{:,.2%}".format,
        "MOTP": "{:,.2%}".format,
        "IDF1": "{:,.2%}".format,
    }

    result = pd.DataFrame([results[c] for c in columns], columns=columns)
    result.fillna(0, inplace=True)

    averages = result.mean().to_frame().T
    averages.rename(index={0: "Average"}, inplace=True)

    result = result.style.format(formatter_dict).data

    logger.log_metrics({"Tracking/MOT": result})
    logger.log_metrics({"Tracking/MOT_Averages": averages})


IOU_THRESHOLD = 0.5
FRAME_BUFFERS = (0, 1, 2)

COUNT_KEYS = (
    "Total GT Divisions",
    "Total Predicted Divisions",
    "True Positive Divisions",
    "False Positive Divisions",
    "False Negative Divisions",
    "Wrong Children Divisions",
)


def format_record(record: dict[str, Any], buffer: int = 0) -> str:
    """One-line summary for a single sequence, for printing during processing."""
    r = _rates(record["counts"][buffer])
    f = lambda v: (
        "n/a" if isinstance(v, float) and math.isnan(v) else f"{v:.4f}"
    )  # noqa: E731
    return (
        f"Divisions [{record['sequence']}] i={buffer}: "
        f"Prec={f(r['Precision'])} Rec={f(r['Recall'])} F1={f(r['F1'])} "
        f"Assoc={f(r['Association Accuracy'])} ({r['TP']}/{r['Association Denominator']}) | "
        f"TP={r['TP']} FP={r['FP']} FN={r['FN']} WC={r['WC']} "
        f"nGT={r['nGT']} nPred={r['nPred']}"
    )


def _resolve(base: Path, track_filename: str) -> tuple[Path, Path]:
    """Accept both ``X_GT/man_track.txt`` and ``X_GT/TRA/man_track.txt``."""
    for candidate in (Path(base), Path(base) / "TRA"):
        track_file = candidate / track_filename
        if track_file.is_file():
            return candidate, track_file
    raise FileNotFoundError(f"No {track_filename} under {base}")


def evaluate_dirs(
    gt_dir: str | Path,
    res_dir: str | Path,
    iou_threshold: float = IOU_THRESHOLD,
    frame_buffers: tuple[int, ...] = FRAME_BUFFERS,
    name: str = "seq",
) -> dict[str, Any]:
    """Division metrics for one sequence, given its GT and RES directories.

    Returns raw counts per frame buffer plus matcher diagnostics. Rates are
    deliberately NOT aggregated here -- pool the counts across sequences with
    ``pool()`` instead, since averaging per-sequence F1 is wrong (see pool()).
    """
    gt_data_dir, gt_track = _resolve(Path(gt_dir), "man_track.txt")
    res_data_dir, res_track = _resolve(Path(res_dir), "res_track.txt")

    gt = load_ctc_data(gt_data_dir.as_posix(), gt_track.as_posix(), name=f"{name}_GT")
    pred = load_ctc_data(
        res_data_dir.as_posix(), res_track.as_posix(), name=f"{name}_RES"
    )

    matcher = BoxIOUMatcher(iou_threshold=iou_threshold)
    results, _ = run_metrics(
        gt_data=gt,
        pred_data=pred,
        matcher=matcher,
        metrics=[DivisionMetrics(max_frame_buffer=max(frame_buffers))],
    )
    per_buffer = results[0]["results"]

    return {
        "sequence": name,
        "config": {
            "matcher": "BoxIOUMatcher",
            "iou_threshold": iou_threshold,
            "frame_buffers": list(frame_buffers),
            "traccuracy": traccuracy.__version__,
        },
        "matching": matcher.diagnostics,
        "counts": {i: per_buffer[f"Frame Buffer {i}"] for i in frame_buffers},
    }


def pool(per_sequence: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Micro-average across sequences: sum counts, then recompute rates.

    Uses traccuracy's own definitions --
        recall    = TP / Total GT Divisions
        precision = TP / Total Predicted Divisions
    The denominators are graph properties, not TP+FN / TP+FP: 'Wrong Children'
    divisions appear in both denominators and in neither numerator.

    Averaging per-sequence F1 would be wrong twice over: a sequence with no
    predicted divisions yields nan, and sequences contribute unequal numbers
    of division events.
    """
    configs = {repr(sorted(r["config"].items())) for r in per_sequence}
    if len(configs) > 1:
        raise ValueError(
            "Sequences were evaluated with different configurations; results are "
            "not comparable. Configs seen: " + repr(configs)
        )

    buffers = sorted(per_sequence[0]["counts"])
    out = {}
    for i in buffers:
        rows = [r["counts"][i] for r in per_sequence]
        totals = {k: sum(int(row[k]) for row in rows) for k in COUNT_KEYS}

        tp = totals["True Positive Divisions"]
        n_gt = totals["Total GT Divisions"]
        n_pred = totals["Total Predicted Divisions"]
        located = tp + totals["Wrong Children Divisions"]

        precision = tp / n_pred if n_pred else math.nan
        recall = tp / n_gt if n_gt else math.nan
        denom = precision + recall
        f1 = (
            2 * precision * recall / denom
            if denom > 0 and not math.isnan(denom)
            else math.nan
        )

        out[i] = {
            **totals,
            "Division Precision": precision,
            "Division Recall": recall,
            "Division F1": f1,
            "Mother-Daughter Association Accuracy": (
                tp / located if located else math.nan
            ),
            "Association Denominator": located,
            "Division Over-Prediction Ratio": n_pred / n_gt if n_gt else math.nan,
            "Num Sequences": len(rows),
        }
    return out


def per_sequence_rows(
    results: list[dict[str, Any]], frame_buffer: int = 0
) -> list[dict[str, Any]]:
    """Flatten per-sequence results into one row per sequence, for a given tolerance.

    Rates use traccuracy's definitions (TP/nPred, TP/nGT). The association
    accuracy is reported together with its denominator, which is small per
    sequence -- treat per-sequence values as noisy.
    """
    rows = []
    for res in results:
        c = res["counts"][frame_buffer]
        tp = c["True Positive Divisions"]
        wc = c["Wrong Children Divisions"]
        n_gt = c["Total GT Divisions"]
        n_pred = c["Total Predicted Divisions"]
        located = tp + wc

        precision = tp / n_pred if n_pred else math.nan
        recall = tp / n_gt if n_gt else math.nan
        denom = precision + recall
        f1 = (
            2 * precision * recall / denom
            if denom > 0 and not math.isnan(denom)
            else math.nan
        )

        rows.append(
            {
                "sequence": res["sequence"],
                "frame_buffer": frame_buffer,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "assoc_accuracy": tp / located if located else math.nan,
                "assoc_numerator": tp,
                "assoc_denominator": located,
                "TP": tp,
                "FP": c["False Positive Divisions"],
                "FN": c["False Negative Divisions"],
                "WC": wc,
                "n_gt_divisions": n_gt,
                "n_pred_divisions": n_pred,
            }
        )
    return rows


def format_table(rows: list[dict[str, Any]], method: str = "method") -> str:
    """Per-sequence table. Compute mean +- std over the rate columns yourself."""
    f = lambda v: (
        "  n/a" if isinstance(v, float) and math.isnan(v) else f"{v:.3f}"
    )  # noqa: E731
    buf = rows[0]["frame_buffer"]
    head = (
        f"{'seq':>6} {'prec':>7} {'rec':>7} {'F1':>7} {'assoc':>7} {'assoc n/d':>11} "
        f"{'TP':>5} {'FP':>5} {'FN':>5} {'WC':>5} {'nGT':>5} {'nPred':>6}"
    )
    lines = [f"{method} -- divisions, tolerance i={buf}", head, "-" * len(head)]
    for r in rows:
        ratio = f"{r['assoc_numerator']}/{r['assoc_denominator']}"
        lines.append(
            f"{r['sequence']:>6} {f(r['precision']):>7} {f(r['recall']):>7} {f(r['f1']):>7} "
            f"{f(r['assoc_accuracy']):>7} {ratio:>11} "
            f"{r['TP']:>5} {r['FP']:>5} {r['FN']:>5} {r['WC']:>5} "
            f"{r['n_gt_divisions']:>5} {r['n_pred_divisions']:>6}"
        )
    return "\n".join(lines)


def write_csv(path, results: list[dict[str, Any]], frame_buffers=FRAME_BUFFERS) -> None:
    """One row per sequence per tolerance, ready for mean +- std in pandas."""
    import csv

    rows = []
    for i in frame_buffers:
        rows.extend(per_sequence_rows(results, i))
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# def append_record(record: dict[str, Any], path: str | Path) -> None:
#     """Append one sequence's result to a JSONL file.

#     For pipelines that evaluate one video per run, so there is no central loop
#     in which to accumulate. Safe to call repeatedly; aggregate later with
#     ``load_records``.
#     """
#     with Path(path).open("a") as fh:
#         fh.write(json.dumps(record, default=str) + "\n")


# def load_records(path: str | Path) -> list[dict[str, Any]]:
#     """Read records back from a JSONL file, de-duplicating by sequence name.

#     Later entries win, so re-running a video overwrites its earlier result
#     rather than double-counting it. Integer frame-buffer keys survive the JSON
#     round-trip as strings, so they are converted back here.
#     """
#     by_sequence: dict[str, dict[str, Any]] = {}
#     with Path(path).open() as fh:
#         for line in fh:
#             line = line.strip()
#             if not line:
#                 continue
#             rec = json.loads(line)
#             rec["counts"] = {int(k): v for k, v in rec["counts"].items()}
#             by_sequence[rec["sequence"]] = rec
#     return [by_sequence[k] for k in sorted(by_sequence)]


def flat_fields(
    record: dict[str, Any],
    frame_buffers: tuple[int, ...] = FRAME_BUFFERS,
    prefix: str = "DIV",
    include_config: bool = False,
) -> dict[str, Any]:
    """Minimal sufficient statistics for one sequence, as DataFrame columns.

    Everything reported later is derived from these::

        precision = TP / nPred
        recall    = TP / nGT
        F1        = 2PR / (P + R)
        assoc     = TP / (TP + WC)

    so only the counts are logged. nGT, nPred and WC do not vary with the
    tolerance (traccuracy holds them fixed across frame buffers and recomputes
    only TP/FP/FN), so they are emitted once; TP is emitted per buffer. FP and
    FN are omitted because they are recoverable: FP = nPred - TP - WC and
    FN = nGT - TP - WC.

    Use ``add_rates`` to turn the logged columns into the reported ones.
    """
    counts = record["counts"]
    first = counts[min(counts)]

    out: dict[str, Any] = {
        f"{prefix}_nGT": int(first["Total GT Divisions"]),
        f"{prefix}_nPred": int(first["Total Predicted Divisions"]),
        f"{prefix}_WC": int(first["Wrong Children Divisions"]),
    }
    for i in frame_buffers:
        if i in counts:
            out[f"{prefix}_TP_i{i}"] = int(counts[i]["True Positive Divisions"])

    if include_config:
        cfg = record.get("config", {})
        out[f"{prefix}_iou_threshold"] = cfg.get("iou_threshold")
        out[f"{prefix}_traccuracy"] = cfg.get("traccuracy")
    return out


def add_rates(df, frame_buffers: tuple[int, ...] = FRAME_BUFFERS, prefix: str = "DIV"):
    """Derive the reported rates from the logged counts, in place on a copy.

    Adds ``{prefix}_Prec_i{n}``, ``_Rec_i{n}``, ``_F1_i{n}`` and ``_Assoc_i{n}``
    for each tolerance. Take mean +- std over the resulting columns.
    """
    df = df.copy()
    n_gt = df[f"{prefix}_nGT"]
    n_pred = df[f"{prefix}_nPred"]
    wc = df[f"{prefix}_WC"]

    for i in frame_buffers:
        tp_col = f"{prefix}_TP_i{i}"
        if tp_col not in df:
            continue
        tp = df[tp_col]
        precision = tp / n_pred.where(n_pred > 0)
        recall = tp / n_gt.where(n_gt > 0)
        located = tp + wc

        df[f"{prefix}_Prec_i{i}"] = precision
        df[f"{prefix}_Rec_i{i}"] = recall
        df[f"{prefix}_F1_i{i}"] = (
            2
            * precision
            * recall
            / (precision + recall).where((precision + recall) > 0)
        )
        df[f"{prefix}_Assoc_i{i}"] = tp / located.where(located > 0)
    return df


def _rates(counts: dict[str, Any]) -> dict[str, Any]:
    tp = int(counts["True Positive Divisions"])
    wc = int(counts["Wrong Children Divisions"])
    n_gt = int(counts["Total GT Divisions"])
    n_pred = int(counts["Total Predicted Divisions"])
    located = tp + wc

    precision = tp / n_pred if n_pred else math.nan
    recall = tp / n_gt if n_gt else math.nan
    denom = precision + recall
    f1 = (
        2 * precision * recall / denom
        if denom > 0 and not math.isnan(denom)
        else math.nan
    )

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Association Accuracy": tp / located if located else math.nan,
        "Association Denominator": located,
        "TP": tp,
        "FP": int(counts["False Positive Divisions"]),
        "FN": int(counts["False Negative Divisions"]),
        "WC": wc,
        "nGT": n_gt,
        "nPred": n_pred,
    }


def per_sequence_rates(
    per_sequence: list[dict[str, Any]], buffer: int = 0
) -> dict[str, dict[str, Any]]:
    return {r["sequence"]: _rates(r["counts"][buffer]) for r in per_sequence}


def format_per_sequence(
    per_sequence: list[dict[str, Any]], buffer: int = 0, method: str = "method"
) -> str:
    rows = per_sequence_rates(per_sequence, buffer)
    f = lambda v: (
        "n/a".rjust(8)
        if isinstance(v, float) and math.isnan(v)
        else f"{v:.4f}".rjust(8)
    )

    head = (
        f"{'video':>6} {'Prec.':>8} {'Rec.':>8} {'F1':>8} {'Assoc.':>8} "
        f"{'(TP/loc)':>9} {'TP':>4} {'FP':>5} {'FN':>4} {'WC':>4} {'nGT':>5} {'nPred':>6}"
    )
    lines = [
        f"{method} -- divisions per video, tolerance i = {buffer}",
        head,
        "-" * len(head),
    ]
    for seq, r in rows.items():
        frac = f"{r['TP']}/{r['Association Denominator']}"
        lines.append(
            f"{seq:>6} {f(r['Precision'])} {f(r['Recall'])} {f(r['F1'])} "
            f"{f(r['Association Accuracy'])} {frac:>9} "
            f"{r['TP']:>4} {r['FP']:>5} {r['FN']:>4} {r['WC']:>4} {r['nGT']:>5} {r['nPred']:>6}"
        )
    return "\n".join(lines)


def write_csv(
    per_sequence: list[dict[str, Any]],
    path: str | Path,
    method: str = "method",
    frame_buffers: tuple[int, ...] = FRAME_BUFFERS,
) -> None:

    fields = [
        "method",
        "video",
        "tolerance",
        "Precision",
        "Recall",
        "F1",
        "Association Accuracy",
        "Association Denominator",
        "TP",
        "FP",
        "FN",
        "WC",
        "nGT",
        "nPred",
    ]
    with Path(path).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i in frame_buffers:
            for seq, r in per_sequence_rates(per_sequence, i).items():
                writer.writerow({"method": method, "video": seq, "tolerance": i, **r})


def format_pooled(pooled: dict[int, dict[str, Any]], method: str = "method") -> str:
    f = lambda v: (
        "n/a" if isinstance(v, float) and math.isnan(v) else f"{v:.4f}"
    )  # noqa: E731
    lines = [
        f"Pooled division metrics ({method}, {pooled[min(pooled)]['Num Sequences']} sequences)"
    ]
    for i, r in sorted(pooled.items()):
        lines.append(
            f"  i={i}: BC={f(r['Division F1'])}  P={f(r['Division Precision'])}  "
            f"R={f(r['Division Recall'])}  "
            f"TP={r['True Positive Divisions']} FP={r['False Positive Divisions']} "
            f"FN={r['False Negative Divisions']} WC={r['Wrong Children Divisions']}"
        )
    r0 = pooled[min(pooled)]
    lines.append(
        f"  mother-daughter association accuracy = "
        f"{f(r0['Mother-Daughter Association Accuracy'])} "
        f"({r0['True Positive Divisions']}/{r0['Association Denominator']})"
    )
    lines.append(
        f"  predicted/GT division ratio = {f(r0['Division Over-Prediction Ratio'])} "
        f"({r0['Total Predicted Divisions']}/{r0['Total GT Divisions']}, matcher-independent)"
    )
    return "\n".join(lines)
