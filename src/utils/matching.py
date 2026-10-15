from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial import distance

from ablation.ByteTrack.byte_tracker import BYTETracker, STrack
from ablation.DeepSORT.deepsort import DeepSort

try:
    from ablation.SORT.sort import Sort
except ImportError:
    Sort = None

from utils.misc import (
    Parameters,
    combine_tracks,
    compute_score,
    get_prev_tracklet,
    get_shared_candidates,
    initialize_track,
    update_patience_dict,
)
from utils.tracking_utils import Cell, Tracklet


class BaseTrackerWrapper:
    def __init__(self, img_info=[1024, 1024], img_size=[1024, 1024]):
        self.tracker = None
        self.img_info = img_info
        self.img_size = img_size

        self.results = []

    def convert_format(
        self, high_score_output, low_score_output, return_embeddings=False
    ):
        """
        Converts the format of our detections to the format used by ByteTrack etc.

        Parameters
        ----------
        :param high_score_output: Dictionary of high confidence detections.
        :param low_score_output: Dictionary of low confidence detections.
        :return: Array holding the combined detections.
        """
        dets = []
        embeddings = []

        for cell in high_score_output.values():
            x1, y1, x2, y2 = cell.bounding_box
            score = float(cell.confidence_score)
            dets.append([x1, y1, x2, y2, score])
            if return_embeddings:
                embeddings.append(cell.embedding)

        for cell in low_score_output.values():
            x1, y1, x2, y2 = cell.bounding_box
            score = float(cell.confidence_score)
            dets.append([x1, y1, x2, y2, score])
            if return_embeddings:
                embeddings.append(cell.embedding)

        if return_embeddings:
            embeddings = torch.stack(embeddings)
            embeddings = embeddings / embeddings.norm(p=2, dim=1, keepdim=True)

            return np.array(dets, dtype=np.float32), embeddings.cpu().numpy()

        return np.array(dets, dtype=np.float32)

    def convert_for_sort(self, dets):
        if len(dets) == 0:
            return torch.empty((0, 6))

        dets = torch.from_numpy(dets)

        obj_conf = dets[:, 4:5]
        class_conf = torch.ones_like(obj_conf)

        return torch.cat([dets[:, :4], obj_conf, class_conf], dim=1)

    def update_tracks(self, time_step, detection: STrack):
        _id = detection.track_id
        bbox = detection.tlbr.tolist()
        centroid = [int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)]
        score = detection.score

        cell = Cell(
            id=_id,
            time_step=time_step,
            centroid=centroid,
            bounding_box=bbox,
            confidence_score=score,
        )

        if len(self.results) == 0:
            # intialize new Tracklet
            tracklet = Tracklet(_id, time_step)
            tracklet.add_cell(time_step, cell)
            self.results.append(tracklet)
        else:
            prev_tracklet = [
                tracklet for tracklet in self.results if tracklet.tracklet_id == _id
            ]

            if len(prev_tracklet) == 0:
                # no track for cell, initialize new one
                tracklet = Tracklet(_id, time_step)
                tracklet.add_cell(time_step, cell)
                self.results.append(tracklet)
            elif len(prev_tracklet) == 1:
                # cell already known, update tracklet
                prev_tracklet = prev_tracklet[0]
                prev_tracklet.propagate_cell(
                    time_step=time_step,
                    prev_label=_id,
                    candidate=cell,
                    recover=False,
                    use_kalman_filter=self.use_interpolation,
                )
            else:
                raise Exception


class BYTETrackMatcher(BaseTrackerWrapper):
    def __init__(
        self,
        track_thresh=0.45,
        track_buffer=5,
        match_thresh=0.8,
        mot20=False,
        fram_rate=30,
        min_box_area=100,
    ):
        super().__init__()

        self.tracker = BYTETracker(
            SimpleNamespace(
                track_thresh=track_thresh,
                track_buffer=track_buffer,
                match_thresh=match_thresh,
                mot20=mot20,
                frame_rate=fram_rate,
                min_box_area=min_box_area,
            )
        )

        self.use_interpolation = False

    def __call__(self, time_step, hs_curr_centroids, ls_curr_centroids):
        dets = self.convert_format(hs_curr_centroids, ls_curr_centroids)

        online_targets = self.tracker.update(dets, self.img_info, self.img_size)

        for t in online_targets:
            tlwh = t.tlwh
            size = tlwh[2] * tlwh[3]

            if size > self.tracker.args.min_box_area:
                self.update_tracks(time_step, t)


class SORTTrackAdapter:
    def __init__(self, arr, score=1.0):
        self._arr = arr

        self.track_id = int(arr[4])
        self.score = float(score)

        x1, y1, x2, y2 = arr[:4]

        self.tlbr = np.array([x1, y1, x2, y2])
        self.tlwh = np.array([x1, y1, x2 - x1, y2 - y1])


class SORTTrackMatcher(BaseTrackerWrapper):
    def __init__(
        self,
        track_thresh=0.45,
        max_age=5,
        min_hits=1,
        min_box_area=100,
    ):
        super().__init__()
        self.track_thresh = track_thresh
        self.min_box_area = min_box_area

        try:
            self.tracker = Sort(max_age=max_age, min_hits=min_hits)
        except TypeError:
            raise ImportError("SORT is not installed. Please install it.")

        self.use_interpolation = False

    def __call__(self, time_step, hs_curr_centroids, ls_curr_centroids):
        dets = self.convert_format(hs_curr_centroids, ls_curr_centroids)
        dets = self.convert_for_sort(dets)

        online_targets = self.tracker.update(
            self._to_sort_dets(dets, self.img_info, self.img_size, self.track_thresh)
        )

        for t in online_targets:
            t = SORTTrackAdapter(t)

            tlwh = t.tlwh
            size = tlwh[2] * tlwh[3]

            if size > self.min_box_area:
                self.update_tracks(time_step, t)

    def _to_sort_dets(self, output_results, img_info, img_size, det_thresh):
        """Convert detector output to SORT's [x1,y1,x2,y2,score] format."""
        output_results = output_results.cpu().numpy()
        scores = output_results[:, 4] * output_results[:, 5]
        bboxes = output_results[:, :4]
        scale = min(img_size[0] / float(img_info[0]), img_size[1] / float(img_info[1]))
        bboxes /= scale
        dets = np.concatenate((bboxes, scores[:, None]), axis=1)
        return dets[scores > det_thresh]


class DeepSORTTrackMatcher(BaseTrackerWrapper):
    def __init__(
        self,
        max_dist=0.1,
        track_thresh=0.3,
        max_iou_distance=0.7,
        max_age=5,
        n_init=1,
        nn_budget=100,
        min_box_area=100,
    ):
        super().__init__()

        self.min_box_area = min_box_area
        self.use_interpolation = False

        try:
            self.tracker = DeepSort(
                max_dist=max_dist,
                min_confidence=track_thresh,
                max_iou_distance=max_iou_distance,
                max_age=max_age,
                n_init=n_init,
                nn_budget=nn_budget,
            )
        except ImportError:
            raise ImportError("DeepSORT is not installed. Please install it.")

    def __call__(self, time_step, hs_curr_centroids, ls_curr_centroids):
        dets, embeddings = self.convert_format(
            hs_curr_centroids, ls_curr_centroids, return_embeddings=True
        )
        dets = self.convert_for_sort(dets)

        online_targets = self.tracker.update(
            dets, embeddings, self.img_info, self.img_size
        )

        for t in online_targets:
            t = SORTTrackAdapter(t)

            tlwh = t.tlwh
            size = tlwh[2] * tlwh[3]

            if size > self.min_box_area:
                self.update_tracks(time_step, t)


def matching_TT(
    time_step,
    hs_curr_centroids: dict,
    ls_curr_centroids: dict,
    prev_centroids: dict,
    last_new_label_index: int,
    tracklets: list,
    patience_dict: dict,
    mitosis: dict,
    patience_range: int,
):
    """
    Matches the cells of the current frame with the cells of the previous frame and the patience dict.
    If a cell is not matched, it is added to the patience dict. If a cell is matched, it is removed from the patience dict.

    Parameters
    ----------
    time_step : int
        The current time step.
    hs_curr_centroids : dict
        Contains information about the cells of the current frame with high confidence scores.
    ls_curr_centroids : dict
        Contains information about the cells of the current frame with low confidence scores.
    prev_centroids : dict
        Contains information about the cells of the previous frame.
    last_new_label_index : int
        The last new label index.
    tracklets : list
        List of all tracklets.
    patience_dict : dict
        Contains lost/unmatched cells from previous frames.
    mitosis : dict
        Contains information about the cells that underwent mitosis.

    Returns
    -------
    list
        Updated list of tracklets.
    int
        Updated last new label index.
    dict
        Updated dictionary of segments for the current frame.
    dict
        Updated patience dictionary.
    dict
        Updated dictionary of mitosis information.
    """

    parameters = Parameters(
        lmbda=0.5,
        sim_threshold=65,
        dist_threshold=50,
        patience_range=patience_range,
        kalman_filter=True,
        low_conf=True,
    )

    curr_segments = dict()

    # First, try to find a matching cell for each cell in the last frame
    candidates = determine_candidates(
        hs_curr_centroids,
        prev_centroids,
        patience_dict,
        parameters,
        time_step,
    )

    # sort dict such that the cell with the lower id's are compared first
    candidates = dict(sorted(candidates.items()))

    # It may happen, that a specific cell of the current frame is associated to two targets of the previous frame. In this case, we keep the assignment with the minimum cost regarding the similarity distance and euclidean distance. The other assignment is deleted.
    shared_candidates = get_shared_candidates(candidates)

    delete_duplicate_candidates(candidates, shared_candidates, parameters)

    # try to associate each cell with its potential candidates that were found in the previous step
    curr_mitosis, prev_matched, curr_matched = associate_candidates(
        candidates,
        tracklets,
        patience_dict,
        time_step,
        prev_centroids,
        curr_segments,
        parameters,
    )

    prev_unmatched = list(set(prev_centroids.keys()) - set(prev_matched))
    curr_unmatched = list(set(hs_curr_centroids.keys()) - set(curr_matched))

    if parameters.use_low_conf():
        prev_unmatched_centroids = {
            key: value for key, value in prev_centroids.items() if key in prev_unmatched
        }

        # try to match the unmatched cells with the low-score detections
        if len(prev_unmatched) > 0 and len(ls_curr_centroids) > 0:
            ls_candidates = determine_candidates(
                ls_curr_centroids,
                prev_unmatched_centroids,
                patience_dict,
                parameters,
                time_step,
            )

            # sort dict such that the cell with the lower id's are compared first
            ls_candidates = dict(sorted(ls_candidates.items()))

            # It may happen, that a specific cell of the current frame is associated to two targets of the previous frame. In this case, we keep the assignment with the minimum cost regarding the similarity distance and euclidean distance. The other assignment is deleted.

            shared_candidates = get_shared_candidates(ls_candidates)

            delete_duplicate_candidates(ls_candidates, shared_candidates, parameters)

            # try to associate each cell with its potential candidates that were found in the previous step
            ls_mitosis, prev_matched_ls, _ = associate_candidates(
                ls_candidates,
                tracklets,
                patience_dict,
                time_step,
                prev_centroids,
                curr_segments,
                parameters,
            )

            curr_mitosis = curr_mitosis | ls_mitosis

            prev_unmatched = list(set(prev_unmatched) - set(prev_matched_ls))

    # for every new cell with a high confidence score in the current frame, create a new tracklet with a new cell
    for label in curr_unmatched:
        # if ( But only for cells near the border
        #     hs_curr_centroids[label].centroid[0] < boundary
        #     or hs_curr_centroids[label].centroid[0] > IMG_SIZE - boundary
        #     or hs_curr_centroids[label].centroid[1] < boundary
        #     or hs_curr_centroids[label].centroid[1] > IMG_SIZE - boundary
        # ):
        last_new_label_index += 1

        tracklets.append(
            initialize_track(
                last_new_label_index,
                hs_curr_centroids[label],
                curr_segments,
                alive=hs_curr_centroids[label].alive,
            )
        )

    if curr_mitosis:
        for label, prop in curr_mitosis.items():
            rename_dict = dict()

            parent_track = [x for x in tracklets if x.tracklet_id == label][0]

            for labelc in prop["childs"]:
                last_new_label_index += 1
                rename_dict.update({labelc: last_new_label_index})

                if label in patience_dict.keys():
                    prev_alive = patience_dict[label].alive
                else:
                    prev_alive = prev_centroids[label].alive

                if labelc in hs_curr_centroids.keys():
                    cell = hs_curr_centroids[labelc]
                else:
                    cell = ls_curr_centroids[labelc]

                # if not prev_alive and cell.alive:
                #     # TODO: dead cell cannot divide!
                #     raise Exception(
                #         "Cell was not alive before but got alive during mitosis?"
                #     )

                alive = prev_alive and cell.alive

                tracklets.append(
                    initialize_track(
                        last_new_label_index,
                        cell,
                        curr_segments,
                        parent=label,
                        alive=alive,
                        generation=parent_track.generation + 1,
                    )
                )

            childs = []
            for old_label, new_label in rename_dict.items():
                childs.append(new_label)
                curr_mitosis[label]["childs"].remove(old_label)
                curr_mitosis[label]["childs"].append(new_label)

            parent_track.divides(time_step - 1, childs=childs)

    update_patience_dict(
        patience_dict, time_step, parameters, prev_centroids, prev_unmatched
    )

    mitosis = mitosis | curr_mitosis

    return tracklets, last_new_label_index, curr_segments, patience_dict, mitosis


def determine_candidates(
    source: dict,
    target: dict,
    patience_dict: dict,
    parameters: Parameters,
    time_step: int,
):
    """Performs a single matching round between the cells of two frames.

    Parameters
    ----------
    source : dict
        Contains information about the cells of the source frame.
    target : dict
        Contains information about the cells of the target frame.
    patience_dict : dict
        Contains lost/unmatched cells from previous frames.
    parameters : Parameters
        Contains the parameters for the matching process.

    Returns
    -------
    dict
        Contains the matches between the cells of the two frames.
    """

    dist_threshold = parameters.get_dist_threshold()
    sim_threshold = parameters.get_sim_threshold()

    candidates = dict()

    combined = patience_dict | target

    # here, we try to associate each cell of the source target with a cell of the source frame
    for prev_label, prev_prop in combined.items():
        prev_centroid = prev_prop.centroid
        prev_embedding = prev_prop.embedding
        prev_alive = prev_prop.alive
        prev_recovered = prev_prop.recovered

        if prev_recovered:
            # raise Exception("ERROR: Tried to match a cell that was already recovered")
            continue

        for curr_label, curr_prop in source.items():
            curr_centroid = curr_prop.centroid
            curr_embedding = curr_prop.embedding
            curr_alive = curr_prop.alive

            centroid_distance = distance.euclidean(prev_centroid, curr_centroid)

            if centroid_distance > dist_threshold:
                continue

            if not prev_alive and curr_alive:
                continue

            centroid_distance /= dist_threshold

            # compare distance of embeddings
            l1_distance = (
                torch.sum(torch.abs(prev_embedding - curr_embedding))
            ) / sim_threshold

            cell = Cell(
                id=curr_label,
                time_step=time_step,
                centroid=curr_centroid,
                bounding_box=curr_prop.bounding_box,
                confidence_score=curr_prop.confidence_score,
                alive=curr_prop.alive,
                embedding=curr_embedding,
                sim_distance=l1_distance,
                centroid_distance=centroid_distance,
            )

            if prev_label in candidates.keys():
                candidates[prev_label].append(cell)
            else:
                candidates[prev_label] = [cell]

    return candidates


def associate_candidates(
    candidates: dict,
    tracklets: list,
    patience_dict: dict,
    time_step: int,
    prev_centroids: dict,
    curr_segments: dict,
    parameters: Parameters,
):
    """Associates the cells of the previous frame with the potentially identified cells of the current frame and (optionally) recovers cells from the patience dict.

    Parameters
    ----------
    candidates : dict
        Possible matches between cells of the current frame and cells of the previous frame or the patience_dict.
    tracklets : list
        List of all tracklets.
    patience_dict : dict
        Contains lost/unmatched cells from previous frames.
    time_step : int
        The current time step.
    prev_centroids : dict
        Contains information about the cells of the previous frame.
    curr_segments : dict
        Contains information about the cells of the current frame.
    parameters : Parameters
        Contains the parameters for the matching process.

    Returns
    -------
    dict
        Contains information about the cells that underwent mitosis.
    list
        List of labels that were matched in the previous frame.
    list
        List of labels that were matched in the current frame.
    """

    mitosis = dict()
    prev_matched = []
    curr_matched = []

    shared_candidate_dict = get_shared_candidates(candidates)

    to_delete = []

    for shared_candidate, cells in shared_candidate_dict.items():
        if not all(len(candidates[candidate]) == 1 for candidate in cells):
            delete_duplicate_candidates(
                candidates, {shared_candidate: cells}, parameters
            )
            to_delete.append(shared_candidate)

    for key in to_delete:
        shared_candidate_dict.pop(key)

    if shared_candidate_dict:
        for common_candidate, cands in shared_candidate_dict.items():
            # TODO:
            # 1. Try to restrict to only do all of this if all of the shared candidates are in the patience_dict

            # 2. Then, also try to include matching between those previous tracklets. If their embeddings are similar, we can just combine them

            prev_tracklets = [
                get_prev_tracklet(tracklets, time_step, re_identified_object)
                for re_identified_object in cands
            ]
            new_cell = candidates[cands[0]][0]

            combined_track, combined_tracks, unused_tracks = combine_tracks(
                tracklets,
                prev_tracklets,
                new_cell,
                time_step,
                parameters,
            )

            if not combined_track:
                delete_duplicate_candidates(
                    candidates, {common_candidate: cands}, parameters
                )
                continue
            elif parameters.verbose():
                print(
                    "Combined tracklets {}".format(
                        [track.tracklet_id for track in combined_tracks]
                    )
                )

            tracklets.append(combined_track)

            curr_segments[combined_track.tracklet_id] = new_cell

            for re_identified_object in combined_tracks:
                track_id = re_identified_object.tracklet_id
                prev_matched.append(track_id)

                candidates.pop(track_id)

                if track_id in patience_dict.keys():
                    if patience_dict[track_id].recovered:
                        raise Exception(
                            "ERROR: Tried to match a cell that was already recovered"
                        )
                    patience_dict.pop(track_id)

            for unused_track in unused_tracks:
                track_id = unused_track.tracklet_id
                candidates[track_id] = list(
                    filter(lambda x: x.id != common_candidate, candidates[track_id])
                )

            curr_matched.append(common_candidate)

    for labelx, candidatesx in candidates.items():
        # get corresponding tracklet
        prev_tracklet = get_prev_tracklet(tracklets, time_step, labelx)

        if labelx in patience_dict.keys() and patience_dict[labelx].recovered:
            raise Exception("ERROR: Tried to match a cell that was already recovered")

        if len(candidatesx) == 1:
            if candidatesx[0].id in curr_matched:
                raise Exception("ERROR: candidate already matched")
            if labelx in prev_matched:
                raise Exception("ERROR: label already matched")

            # only one candidate found that were not matched before => match them
            if labelx in patience_dict.keys():
                associate_candidate(
                    time_step,
                    labelx,
                    prev_tracklet,
                    candidatesx,
                    0,
                    curr_segments,
                    patience_dict=patience_dict,
                    use_kalman_filter=parameters.use_kalman_filter(),
                )
            else:
                associate_candidate(
                    time_step,
                    labelx,
                    prev_tracklet,
                    candidatesx,
                    0,
                    curr_segments,
                    prev_centroids=prev_centroids,
                )

            prev_matched.append(labelx)
            curr_matched.append(candidatesx[0].id)

        elif len(candidatesx) > 1 and labelx not in patience_dict.keys():
            # multiple candidates were found => assume mitosis
            if labelx in mitosis.keys():
                raise Exception("ERROR: mitosis already in dict")
            elif not prev_tracklet.alive:
                # elif not prev_tracklet.alive or (
                #     time_step >= 20 and len(prev_tracklet) < 20
                # ):  # if cell was dead, it cannot divide => get the best candidate only
                argmin_id, _ = determine_min_candidate(candidatesx, parameters)

                # associate the candidate with the minimum score
                associate_candidate(
                    time_step,
                    labelx,
                    prev_tracklet,
                    candidatesx,
                    argmin_id,
                    curr_segments,
                    prev_centroids=prev_centroids,
                    use_kalman_filter=parameters.use_kalman_filter(),
                )

                prev_matched.append(labelx)
                curr_matched.append(candidatesx[argmin_id].id)

            else:
                tmp_candidates = [
                    candidate.id
                    for candidate in candidatesx
                    if candidate.id not in curr_matched
                ]
                mitosis.update(
                    {
                        labelx: {
                            "time_step": time_step,
                            "childs": tmp_candidates,
                        }
                    }
                )
                if labelx in patience_dict.keys():
                    patience_dict[labelx].recovered = True

                prev_matched.append(labelx)
                curr_matched.extend(tmp_candidates)

    return mitosis, prev_matched, curr_matched


def determine_min_candidate(candidates: dict, parameters: Parameters):
    """Determines the candidate with the minimum score. The score is calculated as a weighted sum of the similarity distance and the Euclidean distance.


    Parameters
    ----------
    candidates : dict
        Contains the candidates for the target cell.
    parameters : Parameters
        Contains the parameters for the matching process.
    """
    min_score = float("inf")
    argmin_id = -1
    for idt, cell in enumerate(candidates):
        score = compute_score(cell.sim_distance, cell.centroid_distance, parameters)

        if score < min_score:
            min_score = score
            argmin_id = idt

    return argmin_id, min_score


def associate_candidate(
    time_step: int,
    target_label: int,
    tracklet: Tracklet,
    candidates: dict,
    selected_id: int,
    segments: dict,
    patience_dict: dict = None,
    prev_centroids: dict = None,
    use_kalman_filter: bool = False,
):
    """_summary_

    Parameters
    ----------
    time_step : int
        _description_
    target_label : int
        _description_
    tracklet : Tracklet
        _description_
    candidates : dict
        _description_
    selected_id : int
        _description_
    segments : dict
        _description_
    patience_dict : dict, optional
        _description_, by default None
    prev_centroids : dict, optional
        _description_, by default None
    use_kalman_filter : bool, optional
        _description_, by default False

    Raises
    ------
    Exception
        _description_
    """

    if patience_dict:
        # mark cell as recovered
        if patience_dict[target_label].recovered:
            raise Exception("ERROR: Tried to match a cell that was already recovered")
        patience_dict[target_label].recovered = True

    if prev_centroids:
        alive = candidates[selected_id].alive and prev_centroids[target_label].alive

    else:
        alive = candidates[selected_id].alive

    tracklet.propagate_cell(
        time_step=time_step,
        prev_label=target_label,
        candidate=candidates[selected_id],
        recover=True if patience_dict else False,
        use_kalman_filter=use_kalman_filter,
    )

    # add cell to curr_segments
    segments[target_label] = candidates[selected_id]
    segments[target_label].alive = alive


def delete_duplicate_candidates(
    proposed_dict: dict,
    shared_candidate_dict: dict,
    parameters: Parameters,
):
    """_summary_

    Parameters
    ----------
    proposed_dict : dict
        Contains the proposed matches between cells.
    sens_parameters : list
        List of sensitive parameters use for matching. Contains the similarity threshold, distance threshold, alpha, beta, and the window size.
    """
    for candidate_label, targets in shared_candidate_dict.items():
        min_score = float("inf")
        argmin_id = -1

        for idt, target in enumerate(targets):
            candidates_dict = {t.id: t for t in proposed_dict[target]}

            score = compute_score(
                candidates_dict[candidate_label].sim_distance,
                candidates_dict[candidate_label].centroid_distance,
                parameters,
            )

            if score < min_score:
                min_score = score
                argmin_id = idt
        targets.pop(argmin_id)

        for target in targets:
            indice = [
                proposed_dict[target].index(x)
                for x in proposed_dict[target]
                if x.id == candidate_label
            ][0]
            del proposed_dict[target][indice]

    keys_to_delete = []

    for key, value in proposed_dict.items():
        if not value:
            keys_to_delete.append(key)

    for key in keys_to_delete:
        del proposed_dict[key]
