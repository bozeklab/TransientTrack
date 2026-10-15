import os
import stat
from collections import defaultdict
from itertools import combinations
from typing import List

import torch
import torch.nn.functional as F
from scipy.spatial import distance

from utils.tracking_utils import Cell, Tracklet


class Parameters:
    def __init__(
        self,
        sim_threshold=65,
        dist_threshold=50,
        lmbda=0.5,
        patience_range=5,
        kalman_filter=True,
        low_conf=True,
        verbose_inf=False,
    ):
        self.sim_threshold = sim_threshold
        self.dist_threshold = dist_threshold
        self.alpha = lmbda
        self.beta = 1 - lmbda
        self.patience_range = (
            patience_range if kalman_filter else 0
        )  # necessary to prevent holes in trajectories for CTC evaluation
        self.kalman_filter = kalman_filter
        self.low_conf = low_conf
        self.verbose_inf = verbose_inf

    def get_sim_threshold(self):
        return self.sim_threshold

    def get_dist_threshold(self):
        return self.dist_threshold

    def get_alpha(self):
        return self.alpha

    def get_beta(self):
        return self.beta

    def get_patience_range(self):
        return self.patience_range

    def use_kalman_filter(self):
        return self.kalman_filter

    def use_low_conf(self):
        return self.low_conf

    def verbose(self):
        return self.verbose_inf


def update_patience_dict(
    patience_dict: dict, time_step, parameters: Parameters, prev_centroids, unmatched
):
    """Updates the given patience_dict according to a given time range and the current time step.

    Parameters
    ----------
    patience_dict : dict
        Contains lost/unmatched cells from previous frames.
    time_step : int
        The current time step.
    patience_range : int
        The time range to consider.
    """

    for label in unmatched:
        if label not in patience_dict.keys():
            prev_centroids[label].last_seen = time_step - 1
            patience_dict.update({label: prev_centroids[label]})

    to_delete = [
        label
        for label in patience_dict.keys()
        if abs(patience_dict[label].last_seen - time_step)
        > parameters.get_patience_range()
        or patience_dict[label].recovered
    ]

    for label in to_delete:
        del patience_dict[label]


def get_shared_candidates(proposed_dict):
    shared_candidate_dict = {}
    for key, value in proposed_dict.items():
        for child in value:
            if child.id in shared_candidate_dict:
                shared_candidate_dict[child.id].append(key)
            else:
                shared_candidate_dict[child.id] = [key]

    shared_candidate_dict = {
        k: v for k, v in shared_candidate_dict.items() if len(v) > 1
    }

    return shared_candidate_dict


def get_prev_tracklet(tracklets, time_step, label):
    prev_tracklet = [
        tracklet for tracklet in tracklets if tracklet.tracklet_id == label
    ]

    assert len(prev_tracklet) == 1, f"ERROR at time step {time_step}"

    return prev_tracklet[0]


def is_non_overlapping(subset):
    subset = sorted(subset, key=lambda obj: obj.t_start)
    for i in range(len(subset) - 1):
        if subset[i].t_end > subset[i + 1].t_start:
            return False
    return True


def check_combination(subset, verbose):
    # sort subset by t_start
    subset = sorted(subset, key=lambda obj: obj.t_start)

    messages = ["\nFound possible combination:"]
    for track in subset:
        messages += [
            f"Tracklet {track.tracklet_id} start: {track.t_start} end: {track.t_end}"
        ]
    if verbose:
        for message in messages:
            print(message)

    messages = []
    if not is_non_overlapping(subset):
        messages = ["Tracklets are overlapping"]
    elif not all(track.parent is None for track in subset[1:]):
        messages = ["Tracklets have different parents"]

    # elif not all(track.time_of_death == subset[0].time_of_death for track in subset):
    #     messages = ["Tracklets have different time_of_death values"]
    #     messages += [
    #         f"Tracklet {track.tracklet_id} time_of_death: {track.time_of_death}"
    #         for track in subset
    #     ]
    # elif not all(track.parent == subset[0].parent for track in subset):
    #     messages = ["Tracklets have different parent values"]
    #     messages += [
    #         f"Tracklet {track.tracklet_id} parent: {track.parent}" for track in subset
    #     ]

    if messages:
        if verbose:
            for message in messages:
                print(message)
        return False

    return True


def compute_score(sim_distance, eucl_distance, parameters: Parameters):
    return sim_distance * parameters.get_alpha() + eucl_distance * parameters.get_beta()


def compute_similarity_score(emb1, emb2):
    return torch.sum(torch.abs(emb1 - emb2))


def compute_cosine_similarity(emb1, emb2):
    return 1 - F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0), dim=-1)


def compute_euclidean_distance(centroid1, centroid2):
    return distance.euclidean(centroid1, centroid2)


def pairwise_similarity_avg(combination: List[Tracklet]):
    n = len(combination)
    score = sum(
        compute_cosine_similarity(a.last_embedding, b.last_embedding)
        for a, b in combinations(combination, 2)
    )
    num_pairs = n * (n - 1) // 2

    return score / num_pairs


def pairwise_euclidean_distance_avg(combination: List[Tracklet]):
    n = len(combination)
    score = sum(
        compute_euclidean_distance(a.get_last_position(), b.get_first_position())
        for a, b in combinations(combination, 2)
    )
    num_pairs = n * (n - 1) // 2

    return score / num_pairs


def priority_min_with_threshold(combinations, sim_threshold):
    grouped_by_length = defaultdict(list)
    for combination in combinations:
        grouped_by_length[len(combination)].append(combination)

    sorted_lengths = sorted(grouped_by_length.keys(), reverse=True)

    for length in sorted_lengths:
        tuple_of_this_length = grouped_by_length[length]

        min_combination, min_combination_score = min(
            (
                (combination, pairwise_similarity_avg(combination))
                for combination in tuple_of_this_length
            ),
            key=lambda x: x[1],
        )
        if min_combination_score < sim_threshold:
            return min_combination, min_combination_score

    return None, None


def combine_tracks(
    tracklets,
    curr_tracks,
    curr_candidate,
    time_step,
    parameters: Parameters,
):
    sim_threshold = parameters.get_sim_threshold()

    # sanity checks
    curr_tracks.sort(key=lambda x: x.t_start)

    all_combinations = []
    for r in range(2, len(curr_tracks) + 1):
        all_combinations.extend(combinations(curr_tracks, r))

    valid_combinations = [
        list(subset)
        for subset in all_combinations
        if check_combination(subset, parameters.verbose())
    ]

    if not valid_combinations:
        return False, None, None

    if len(valid_combinations) == 1:
        min_combination = valid_combinations[0]
        min_combination_score = pairwise_similarity_avg(min_combination)
    elif len(valid_combinations) > 1:
        if parameters.verbose():
            messages = ["\nMore than 1 valid combination\n"]
            print(messages)
            messages += [
                f"Tracklet {track.tracklet_id} start: {track.t_start} end: {track.t_end}"
                for track in curr_tracks
            ]
            for message in messages:
                print(message)
            print("\n")

        min_combination, min_combination_score = priority_min_with_threshold(
            valid_combinations, sim_threshold
        )

    if min_combination_score and min_combination_score < sim_threshold:
        valid_combination = min_combination
        if parameters.verbose():
            print(f"Combination score: {min_combination_score}")
    else:
        if parameters.verbose():
            print(
                f"Combination scores too high{f': {min_combination_score}' if min_combination_score else ''}"
            )
        return False, None, None

    unused = list(set(curr_tracks) - set(valid_combination))

    # get the minimum id and the time step of the first cell of the tracklet
    min_id = min([track.tracklet_id for track in valid_combination])
    min_t_start = min([track.t_start for track in valid_combination])
    max_t_end = max(max([track.t_end for track in valid_combination]), time_step)

    # create a new tracklet with the minimum id and the time step of the first cell of the tracklet
    track = Tracklet(min_id, min_t_start, parent=valid_combination[0].parent)

    start_cell = max(
        [
            track.get_cell(min_t_start)
            for track in valid_combination
            if track.get_cell(min_t_start) is not None
        ],
        key=lambda x: x.confidence_score,
    )

    start_cell.id = min_id
    track.add_cell(min_t_start, start_cell)

    # add the cells of the combined tracklets with the highest confidence score to the new tracklet
    hole = False
    for i in range(min_t_start + 1, max_t_end + 1):
        cell = max(
            [
                track.get_cell(i)
                for track in valid_combination
                if track.get_cell(i) is not None
            ],
            key=lambda x: x.confidence_score,
            default=None,
        )
        if cell is None:
            hole = True
            continue
        track.propagate_cell(
            time_step=i,
            prev_label=min_id,
            candidate=cell,
            recover=hole,
            use_kalman_filter=parameters.use_kalman_filter(),
        )
        hole = False

    # add the new cell to the new tracklet
    recover = True if max_t_end != track.t_end else False
    track.propagate_cell(
        time_step=time_step,
        prev_label=min_id,
        candidate=curr_candidate,
        recover=recover,
        use_kalman_filter=parameters.use_kalman_filter(),
    )

    # delete the combined tracklets from the list of tracklets
    for old_track in valid_combination:
        tracklets.remove(old_track)

    return track, valid_combination, unused


def initialize_track(
    label,
    cell,
    curr_centroids,
    parent=None,
    alive=True,
    generation=0,
):
    tracklet = Tracklet(label, cell.time_step, parent=parent, generation=generation)

    cell = Cell(
        id=label,
        time_step=cell.time_step,
        centroid=cell.centroid,
        bounding_box=cell.bounding_box,
        confidence_score=cell.confidence_score,
        embedding=cell.embedding,
        alive=alive,
    )

    tracklet.add_cell(cell.time_step, cell)

    curr_centroids[label] = cell

    return tracklet


def write_tracks(tracklets, output_dir, start_id=49, end_id=185):
    with open(os.path.join(output_dir, "tracks.csv"), "a") as file:
        for tracklet in tracklets:
            end = tracklet.t_end if tracklet.t_end is not None else end_id - 1
            parent = tracklet.parent if tracklet.parent is not None else 0
            time_of_division = (
                tracklet.time_of_division
                if tracklet.time_of_division is not None
                else 0
            )
            time_of_death = (
                tracklet.time_of_death if tracklet.time_of_death is not None else 0
            )
            generation = tracklet.generation
            file.write(
                f"{tracklet.tracklet_id};{tracklet.t_start};{end};{time_of_division};{time_of_death};{parent};{generation}\n"
            )


def remove_detected_objects(remove_list, centroids, masks, labels):
    temp_list_1, temp_list_2, temp_list_3 = [], [], []
    for idx in range(len(labels)):
        if idx in remove_list:
            continue
        temp_list_1.append(centroids[idx])
        temp_list_2.append(masks[idx])
        temp_list_3.append(labels[idx])
    centroids = temp_list_1
    masks = temp_list_2
    labels = temp_list_3
    return centroids, masks, labels


def remove_readonly(func, path, exc):
    if func not in (os.unlink, os.rmdir) or exc[1].winerror != 5:
        raise exc[1]
    os.chmod(path, stat.S_IWRITE)
    func(path)
