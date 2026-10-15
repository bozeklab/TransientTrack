import json
import os

import cv2
import numpy as np
import supervision as sv
import torch
import torchmetrics
from torchvision.ops import nms
from transformers.models.deformable_detr.image_processing_deformable_detr import (
    center_to_corners_format,
)


def calc_bounding_box(mask, id):
    coords = np.where(mask == id)
    if len(coords[0]) == 0:
        print("ERROR with bounding box")
        return None

    y1 = coords[0].min()
    y2 = coords[0].max()

    x1 = coords[1].min()
    x2 = coords[1].max()

    return [x1, y1, x2, y2]


def annotate_image(
    img,
    img_type,
    candidates,
    mitosis,
    label_children,
    corruption=False,
    mean=None,
    std=None,
):
    """Adds labels to an image.

    Parameters
    ----------
    img : ndarray
        The image to which the labels should be added.
    labels : list
        Contains the labels of the cells.
    centroids : list
        Contains the centroids of the cells.

    Returns
    -------
    ndarray
        The image with the added labels.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    text_scale = 0.75
    text_thickness = 1
    text_padding = 2

    col_can_alive = sv.Color.GREEN if img_type == "rgb" else sv.Color.WHITE
    col_can_dead = sv.Color.RED if img_type == "rgb" else sv.Color.WHITE
    col_text = sv.Color.BLACK if img_type == "rgb" else sv.Color.WHITE

    if len(img.shape) > 2:
        img = np.transpose(img, (1, 2, 0))[:, :, ::-1].copy()

    # if mean is not None and std is not None:
    #     mean = np.array(mean)[::-1].reshape(1, 1, 3)
    #     std = np.array(std)[::-1].reshape(1, 1, 3)
    #     img = img * std + mean

    if not corruption:
        if len(img.shape) > 2 and False:
            for c in range(img.shape[2]):
                img[:, :, c] = (img[:, :, c] - np.min(img[:, :, c])) / (
                    np.max(img[:, :, c]) - np.min(img[:, :, c])
                )
                # img[:, :, c] = np.where(
                #     img[:, :, c] >= np.mean(img[:, :, c]),
                #     img[:, :, c] - np.mean(img[:, :, c]),
                #     0,
                # )
        else:
            # mean = np.array(mean).reshape(1, 1, 3)
            # std = np.array(std).reshape(1, 1, 3)
            # img = img * std + mean
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
    else:
        # this is a hacky solution to visualize the corruption, only meant for debugging
        # x = img * 65535.0
        # x = np.log1p(x)
        # x = x / np.log1p(65535)
        # img = x

        img = np.clip(img, -5, 5)
        img = (img + 5) / 10
        # img = (img - np.min(img)) / (np.max(img) - np.min(img))

    img = (img * 255).astype(np.uint8)

    resize_ratio = 2048 / img.shape[0]

    img = cv2.resize(
        img,
        (int(img.shape[0] * resize_ratio), int(img.shape[1] * resize_ratio)),
        interpolation=cv2.INTER_LINEAR,
    )

    for candidate in candidates:
        confidence = candidate.confidence_score
        idx = candidate.id
        bounding_box = candidate.bounding_box
        bounding_box = [int(i * resize_ratio) for i in bounding_box]
        if candidate.alive:
            color = col_can_alive
        else:
            color = col_can_dead

        text_bgr = (
            color.as_rgb() if img_type == "rgb" else sv.Color(127, 127, 127).as_bgr()
        )

        parent_id = [id for id, value in mitosis.items() if idx in value["childs"]]

        if parent_id and label_children:
            label = f"{mitosis[parent_id[0]]['childs'].index(idx) + 1}"
            generation = 0
            while parent_id:
                ancestor = parent_id[0]
                if generation != 0:
                    label = f"{mitosis[ancestor]['childs'].index(parent) + 1}_{label}"
                parent_id = [
                    id for id, value in mitosis.items() if ancestor in value["childs"]
                ]
                parent = ancestor
                generation += 1

            assigned_id = f"{ancestor}_{label}"
        else:
            assigned_id = idx

        x1, y1, x2, y2 = bounding_box

        cv2.rectangle(
            img=img,
            pt1=(x1, y1),
            pt2=(x2, y2),
            color=color.as_rgb(),
            thickness=thickness,
        )

        text = f"id:{assigned_id} {confidence:0.2f}"
        text_width, text_height = cv2.getTextSize(
            text=text, fontFace=font, fontScale=text_scale, thickness=text_thickness
        )[0]

        text_x = x1 + text_padding
        text_y = y1 - text_padding

        text_background_x1 = x1
        text_background_y1 = y1 - 2 * text_padding - text_height

        text_background_x2 = x1 + 2 * text_padding + text_width
        text_background_y2 = y1

        if text_background_y1 < 0:
            text_y = y2 + 8 * text_padding
            text_background_y1 = text_y - text_padding - text_height
            text_background_y2 = text_y + text_padding

        cv2.rectangle(
            img=img,
            pt1=(text_background_x1, text_background_y1),
            pt2=(text_background_x2, text_background_y2),
            color=text_bgr,
            thickness=cv2.FILLED,
        )

        cv2.putText(
            img=img,
            text=text,
            org=(text_x, text_y),
            fontFace=font,
            fontScale=text_scale,
            color=col_text.as_rgb(),
            thickness=text_thickness,
            lineType=cv2.LINE_AA,
        )

    return img


def update_map(
    logits,
    pred_boxes,
    annotations,
    metric: torchmetrics.Metric,
    score_thresh=0,
):
    # TODO: filter out low-confidence boxes?
    probs = logits.sigmoid()
    scores, classes = torch.max(probs, dim=-1)

    preds = []

    for i in range(logits.shape[0]):
        keep = (classes[i] != 0) & (scores[i] > score_thresh)

        pred_labels = classes[i][keep] - 1
        pred_scores = scores[i][keep]
        pred_boxes_i = pred_boxes[i][keep]

        preds.append(
            {
                "boxes": (pred_boxes_i.to("cpu") * 1024),
                "scores": pred_scores.to("cpu"),
                "labels": pred_labels.to("cpu"),
            }
        )

    targets = []

    for i in range(len(annotations)):
        gt_labels = annotations[i]["class_labels"] - 1

        targets.append({"boxes": (annotations[i]["boxes"] * 1024), "labels": gt_labels})

    # preds = [
    #     {
    #         "boxes": (pred_boxes[i][pred_classes[i] != 0].to("cpu") * 1024),
    #         "scores": pred_scores[i][pred_classes[i] != 0].to("cpu"),
    #         "labels": pred_classes[i][pred_classes[i] != 0].to("cpu"),
    #     }
    #     for i in range(logits.shape[0])
    # ]
    # targets = [
    #     {
    #         "boxes": (annotations[i]["boxes"] * 1024),
    #         "labels": annotations[i]["class_labels"],
    #     }
    #     for i in range(len(annotations))
    # ]

    metric.update(preds, targets)


def get_centroids(
    input_image,
    time_step,
    detector,
    device,
    low_conf_higher,
    low_conf_lower,
    iou_threshold,
    metric=None,
    labels=None,
):
    high_score_output = {}
    low_score_output = {}

    with torch.no_grad():
        input_image = input_image.to(device)

        logits, pred_boxes, _, _, embeddings = detector(input_image)

        if metric is not None:
            update_map(logits, pred_boxes, labels, metric, score_thresh=0.1)

        tmp_results = post_process_object_detection(
            logits,
            pred_boxes,
            embeddings,
            low_conf_higher,
            low_conf_lower,
            iou_threshold,
            [(1024, 1024)],
            100,
        )

        results = tmp_results[0]

        if len(results["high_scores"]):
            high_score_detections = sv.Detections(
                xyxy=results["high_boxes"],
                confidence=results["high_scores"],
                class_id=results["high_labels"],
                data={"embeddings": results["high_embeddings"]},
            )
        else:
            high_score_detections = []

        if len(results["low_scores"]):
            low_score_detections = sv.Detections(
                xyxy=results["low_boxes"],
                confidence=results["low_scores"],
                class_id=results["low_labels"],
                data={"embeddings": results["low_embeddings"]},
            )
        else:
            low_score_detections = []

        # detections hold all the results in the following order:
        # 0: bounding box, 1: mask, 2: confidence score, 3: class_id, 4: tracker_id, 5: additional data

        tmp_id = 1
        for (
            bounding_box,
            _,
            confidence_score,
            class_id,
            _,
            embedding,
        ) in high_score_detections:
            b_box = bounding_box.astype(int).tolist()

            centroid = [int((b_box[0] + b_box[2]) / 2), int((b_box[1] + b_box[3]) / 2)]

            high_score_output[tmp_id] = Cell(
                id=tmp_id,
                time_step=time_step,
                centroid=centroid,
                embedding=torch.from_numpy(embedding["embeddings"]),
                confidence_score=confidence_score,
                alive=True if class_id == 1 else False,
                bounding_box=b_box,
            )

            tmp_id += 1

        for (
            bounding_box,
            _,
            confidence_score,
            class_id,
            _,
            embedding,
        ) in low_score_detections:
            b_box = bounding_box.astype(int).tolist()

            centroid = [int((b_box[0] + b_box[2]) / 2), int((b_box[1] + b_box[3]) / 2)]

            low_score_output[tmp_id] = Cell(
                id=tmp_id,
                time_step=time_step,
                centroid=centroid,
                embedding=torch.from_numpy(embedding["embeddings"]),
                confidence_score=confidence_score,
                alive=True if class_id == 1 else False,
                bounding_box=b_box,
            )

            tmp_id += 1

    return high_score_output, low_score_output


def post_process_object_detection(
    logits,
    pred_boxes,
    embeds=None,
    high_conf_lower_bound=0.45,
    low_conf_lower_bound=0.25,
    iou_threshold=0.3,
    target_sizes=None,
    top_k=100,
):
    """Taken and modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/deformable_detr/image_processing_deformable_detr.py#L1496"""

    if target_sizes is not None:
        if len(logits) != len(target_sizes):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the logits"
            )

    prob = logits.sigmoid()
    prob = prob.view(logits.shape[0], -1)
    k_value = min(top_k, prob.size(1))
    topk_values, topk_indexes = torch.topk(prob, k_value, dim=1)
    scores = topk_values
    topk_boxes = torch.div(topk_indexes, logits.shape[2], rounding_mode="floor")
    labels = topk_indexes % logits.shape[2]
    boxes = center_to_corners_format(pred_boxes)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
    topk_embeddings = torch.gather(
        embeds, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 256)
    )

    # and from relative [0, 1] to absolute [0, height] coordinates
    if target_sizes is not None:
        if isinstance(target_sizes, list):
            img_h = torch.Tensor([i[0] for i in target_sizes])
            img_w = torch.Tensor([i[1] for i in target_sizes])
        else:
            img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
        boxes = boxes * scale_fct[:, None, :]

    results = []
    for s, l, b, e in zip(scores, labels, boxes, topk_embeddings):
        high_score = s[s > high_conf_lower_bound]
        high_label = l[s > high_conf_lower_bound]
        high_box = b[s > high_conf_lower_bound]
        high_embeddings = e[s > high_conf_lower_bound]

        mask = (s > low_conf_lower_bound) & (s <= high_conf_lower_bound)
        low_score = s[mask]
        low_label = l[mask]
        low_box = b[mask]
        low_embeddings = e[mask]

        # if len(low_score) > 0 and len(high_score) > 0:
        # keep = nms(b, s, 0.3)
        # keep_score = s[keep].cpu().numpy()
        # keep_label = l[keep].cpu().numpy().astype(int)
        # keep_box = b[keep].cpu().numpy()
        # keep_embeddings = e[keep].cpu().numpy()

        # high_score = keep_score[keep_score > high_conf_lower_bound]
        # high_label = keep_label[keep_score > high_conf_lower_bound]
        # high_box = keep_box[keep_score > high_conf_lower_bound]
        # high_embeddings = keep_embeddings[keep_score > high_conf_lower_bound]

        # mask = (keep_score > low_conf_lower_bound) & (
        #     keep_score <= high_conf_lower_bound
        # )
        # low_score = keep_score[mask]
        # low_label = keep_label[mask]
        # low_box = keep_box[mask]
        # low_embeddings = keep_embeddings[mask]

        # high-confidence NMS
        if len(high_score) > 0:
            keep_high = nms(high_box, high_score, iou_threshold)  # Apply NMS
            high_score = high_score[keep_high].cpu().numpy()
            high_label = high_label[keep_high].cpu().numpy().astype(int)
            high_box = high_box[keep_high].cpu().numpy()
            high_embeddings = high_embeddings[keep_high].cpu().numpy()

        # low-confidence NMS
        if len(low_score) > 0:
            keep_low = nms(
                low_box, low_score, iou_threshold
            )  # Apply NMS for low-confidence boxes
            low_score = low_score[keep_low].cpu().numpy()
            low_label = low_label[keep_low].cpu().numpy().astype(int)
            low_box = low_box[keep_low].cpu().numpy()
            low_embeddings = low_embeddings[keep_low].cpu().numpy()

        # for the low-conf detections, we also keep only the ones that have a low IoU with the high-conf detections
        if len(low_score) > 0 and len(high_score) > 0:
            ious = sv.box_iou_batch(high_box, low_box)

            mask = ~np.any(ious > (1 - iou_threshold), axis=0)
            low_score = low_score[mask]
            low_label = low_label[mask]
            low_box = low_box[mask]
            low_embeddings = low_embeddings[mask]

        results.append(
            {
                "high_scores": high_score,
                "high_labels": high_label,
                "high_boxes": high_box,
                "high_embeddings": high_embeddings,
                "low_scores": low_score,
                "low_labels": low_label,
                "low_boxes": low_box,
                "low_embeddings": low_embeddings,
            }
        )

    return results


def export_summary(tracklets, mitosis, eval_dir):
    tracklets_output = {"summary": len(tracklets), "tracklets": []}
    mitosis_output = {"cell_divisions": len(mitosis), "mitosis": mitosis}

    for tracklet in tracklets:
        tracklets_output["tracklets"].append(tracklet.toJSON())

    with open(os.path.join(eval_dir, "summary.json"), "w") as f:
        output = json.dumps(tracklets_output, cls=ComplexEncoder)
        f.write(output)

    with open(os.path.join(eval_dir, "mitosis.json"), "w") as f:
        output = json.dumps(mitosis_output)
        f.write(output)


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "toJSON"):
            return obj.toJSON()
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float32):
            return obj.item()
        else:
            return json.JSONEncoder.default(self, obj)


class Tracklet:
    """Depicts the information of a whole track with specific possible candidates for each time step."""

    def __init__(
        self,
        tracklet_id,
        t_start,
        t_end=None,
        time_of_death=None,
        parent=None,
        generation=0,
        alive=True,
    ):
        self.tracklet_id = tracklet_id
        self.t_start = t_start
        self.t_end = t_end  # inclusive, points to the last frame in which the cell is apparent or alive
        self.time_of_death = time_of_death  # time of death is inclusive, meaning, it points towards the first frame, in which the cell is dead
        self.time_of_division = None

        self.last_embedding = None

        self.parent = parent
        self.generation = generation
        self.alive = alive
        self.childs = None

        self.candidates = {}

    def divides(self, time_step, childs):
        self.time_of_division = time_step
        self.childs = childs

    def dies(self, time_step):
        self.alive = False
        self.time_of_death = time_step

    def toJSON(self):
        lifespan = (
            self.time_of_death - self.t_start
            if self.time_of_death
            else self.t_end - self.t_start + 1
        )

        if self.time_of_death is not None:
            fate = "apoptosis"
        elif self.time_of_division is not None:
            fate = "mitosis"
        else:
            fate = "alive"

        output = dict(
            {
                "tracklet_id": self.tracklet_id,
                "parent": self.parent,
                "generation": self.generation,
                "tracklet_start": self.t_start,
                "tracklet_end": self.t_end,
                "fate": fate,
                "childs": self.childs,
                "time_of_death": self.time_of_death,
                "time_of_division": self.time_of_division,
                "lifespan": lifespan,
                "frames": [],
            }
        )

        for i in range(self.t_start, self.t_end + 1):
            frame_information = {"frame_id": i, "cell": self.get_cell(i)}
            output["frames"].append(frame_information)

        return output

    @staticmethod
    def fromJSON(json_dict):
        tracklet = Tracklet(
            json_dict["tracklet_id"],
            json_dict["tracklet_start"],
            json_dict["tracklet_end"],
            json_dict["time_of_death"],
            json_dict["parent"],
        )

        frames = json_dict["frames"]
        for frame in frames:
            candidates = []
            frame_id = frame["frame_id"]
            tmp_candidates = frame["cell"]
            for candidate in tmp_candidates:
                cell = Cell.fromJSON(candidate)
                candidates.append(cell)
            tracklet.candidates[frame_id] = candidates

        return tracklet

    def __len__(self):
        length = (
            self.time_of_death - self.t_start
            if self.time_of_death
            else self.t_end - self.t_start
        )

        return length + 1

    def propagate_cell(
        self,
        time_step,
        prev_label,
        candidate,
        recover=False,
        use_kalman_filter=False,
    ):
        # TODO: when merging, only use the conf score of the object
        prev_cell = self.candidates[sorted(self.candidates.keys())[-1]]

        assert (
            prev_cell is not None
        ), "ERROR, could not find exactly one cell while propagating"

        if use_kalman_filter:
            self.fill_gaps(
                time_step,
                prev_cell,
                candidate.centroid,
                candidate.alive,
                candidate.bounding_box,
            )
        else:
            if prev_cell.alive and not candidate.alive:
                self.dies(time_step)

            if not prev_cell.alive:
                candidate.alive = False

        new_cell = Cell(
            id=prev_label,
            time_step=time_step,
            centroid=candidate.centroid,
            bounding_box=candidate.bounding_box,
            confidence_score=candidate.confidence_score,
            alive=candidate.alive,
            embedding=candidate.embedding,
            recovered=recover,
        )

        self.add_cell(time_step, new_cell)

    def fill_gaps(self, time_step, prev_cell, centroid, curr_alive, bounding_box):
        initial_cent = prev_cell.centroid

        # get the newest known bounding box and assume the size of the bounding box is not changing
        bb_w = bounding_box[2] - bounding_box[0]
        bb_h = bounding_box[3] - bounding_box[1]

        # if we have seen the cell only for a single frame so far, we interpolate the position based on the first position and the recovered position instead of using the Kalman filter
        if self.t_start == self.t_end:
            missing_frames = time_step - self.t_end

            steps = [
                (centroid[0] - initial_cent[0]) / missing_frames,
                (centroid[1] - initial_cent[1]) / missing_frames,
            ]

            for j in range(1, missing_frames):
                new_cent = [
                    int(steps[0] + initial_cent[0]),
                    int(steps[1] + initial_cent[1]),
                ]

                new_bb = [
                    int(new_cent[0] - bb_w // 2),
                    int(new_cent[1] - bb_h // 2),
                    int(new_cent[0] + bb_w // 2),
                    int(new_cent[1] + bb_h // 2),
                ]

                new_cell = Cell(
                    id=prev_cell.id,
                    time_step=time_step - missing_frames + j,
                    centroid=new_cent,
                    bounding_box=new_bb,
                    confidence_score=0,
                    alive=curr_alive,
                )
                self.add_cell(time_step - missing_frames + j, new_cell)

                initial_cent = new_cent
        else:
            kalman = KalmanFilter(initial_cent)

            # correct the kalman filter from the beginning of the tracklet until the last occurence of prev_cell
            for k in range(self.t_start, self.t_end + 1):
                candidate = self.get_cell(k)

                kalman.fit(candidate.centroid)

            if not prev_cell.alive:  # make sure that a dead cell stays dead
                curr_alive = False

            if prev_cell.alive and not curr_alive:
                self.dies(self.t_end)

            for k in range(self.t_end + 1, time_step):
                prediction = kalman.predict()

                new_bb = [
                    int(prediction[0] - bb_w // 2),
                    int(prediction[1] - bb_h // 2),
                    int(prediction[0] + bb_w // 2),
                    int(prediction[1] + bb_h // 2),
                ]

                new_cell = Cell(
                    id=prev_cell.id,
                    time_step=k,
                    centroid=prediction.astype(np.int32).flatten()[:-2].tolist(),
                    bounding_box=new_bb,
                    confidence_score=0,
                    alive=curr_alive,
                )
                self.add_cell(k, new_cell)

                prev_cell = new_cell

    def add_cell(self, time_step, cell):
        if time_step not in self.candidates.keys():
            self.candidates[time_step] = cell
        else:
            self.candidates[time_step].append(cell)
            raise Exception("Shouldn't happen")
        if self.t_end is None:
            self.t_end = time_step
        else:
            self.t_end = max(self.t_end, time_step)

        self.last_embedding = cell.embedding

    def get_cell(self, time_step):
        if time_step not in self.candidates.keys():
            return None
        return self.candidates[time_step]

    def get_first_position(self):
        return self.candidates[self.t_start].centroid

    def get_last_position(self):
        return self.candidates[self.t_end].centroid


class Cell:
    """Contains information of a single cell within a specific time step. These information are used as candidates for a whole tracklet."""

    def __init__(
        self,
        id,
        time_step,
        centroid,
        bounding_box,
        confidence_score,
        alive=True,
        embedding=None,
        sim_distance=None,
        centroid_distance=None,
        last_seen=None,
        recovered=False,
        successor=None,
    ):
        self.id = id
        self.time_step = time_step
        self.centroid = centroid
        self.bounding_box = bounding_box
        self.confidence_score = confidence_score
        self.alive = alive
        self.embedding = embedding
        self.sim_distance = sim_distance
        self.centroid_distance = centroid_distance
        self.last_seen = last_seen
        self.recovered = recovered

        self.successor = set()

        if successor is not None:
            self.successor.update(successor)

    def toJSON(self):
        return dict(
            id=self.id,
            time_step=self.time_step,
            centroid=self.centroid,
            confidence_score=self.confidence_score,
            alive=self.alive,
            bounding_box=self.bounding_box,
        )

    @staticmethod
    def fromJSON(json_dict):
        return Cell(
            json_dict["id"],
            json_dict["time_step"],
            json_dict["centroid"],
            json_dict["bounding_box"],
            json_dict["confidence_score"],
            json_dict["alive"],
        )


class KalmanFilter:
    """Implements a Kalman Filter for filling missing objects in a tracklet after recovering a cell."""

    def __init__(self, initial_pos):
        """Initializes the Kalman Filter with the implementation of OpenCV and additionally sets the initial state.

        Parameters
        ----------
        initial_pos : list
            Contains the initial position of the cell's centroid.
        """

        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )

        self.kf.statePre = np.concatenate(
            (initial_pos, [0, 0]), axis=0, dtype=np.float32
        )
        self.kf.statePost = self.kf.statePre

    def fit(self, position):
        """Predicts and then corrects the prediction with the actual measurement.

        Parameters
        ----------
        x : int
            The x-coordinate of the cell's centroid.
        y : int
            The y-coordinate of the cell's centroid.
        """
        # first, do prediction to update the internal statePre variable
        self.kf.predict()
        # then, correct the prediction value with actual measurement
        self.kf.correct(
            np.array([[np.float32(position[0])], [np.float32(position[1])]])
        )

    def predict(self):
        """Predicts the next position of the cell's centroid.

        Returns
        -------
        ndarray
            The predicted position of the cell's centroid.
        """
        return self.kf.predict()
