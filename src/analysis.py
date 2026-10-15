import argparse
import json
import os

import networkx as nx
import numpy as np
import pandas as pd
from pyvis.network import Network
from tqdm import tqdm

from dataset.dataloader import CocoVidDataLoader
from dataset.dataset import InferenceDataset
from utils.analysis_utils import (
    Record,
    combine_figures,
    compute_mean_area,
    compute_size_slope,
    normalize_crop,
    plot_ancestor_descendant,
    plot_division_profile,
    plot_division_timing,
    plot_fate_probability,
    plot_root_descendant,
    plot_sister_cell,
)

doses = {
    "High": list(range(1, 47)),
    "Unknown": list(range(47, 130)) + list(range(202, 274)),
    "Medium": list(range(130, 166)),
    "Low": list(range(166, 202)),
    "Control": list(range(274, 310)),
}


def get_dosage(video):
    for dose, videos in doses.items():
        if int(video) in videos:
            return dose


def add_tracklet_to_graph(G, tracklet, visualize):
    if visualize:
        data = tracklet
        data.update(
            {
                "tracklet_id": tracklet["tracklet_id"],
                "generation": tracklet["generation"],
                "start": tracklet["tracklet_start"],
                "end": tracklet["tracklet_end"],
                "fate": tracklet["fate"],
                "cells": tracklet["frames"],
            }
        )

        if tracklet["fate"] == "mitosis":
            fate_time = tracklet["time_of_division"]
            fate_text = "Time of Division"
            color = "green"
        elif tracklet["fate"] == "apoptosis":
            fate_time = tracklet["time_of_death"]
            fate_text = "Time of Death"
            color = "red"
        else:
            color = "gray"
            fate_time = None

        title = "Start: {}\nEnd: {}\nLifespan: {}".format(
            tracklet["tracklet_start"],
            tracklet["tracklet_end"],
            tracklet["lifespan"],
        )

        if fate_time:
            title += "\n{}: {}".format(fate_text, fate_time)

        G.add_node(
            tracklet["tracklet_id"],
            title=title,
            data=data,
            label=str(tracklet["tracklet_id"]),
            color=color,
        )
    else:
        data = tracklet
        G.add_node(tracklet["tracklet_id"], data=data)


def division_profile_analysis(graph, dosage, pre_treatment):
    roots = []
    division_profile_records = []

    for node_id, data in graph.nodes(data=True):
        graph.nodes[node_id]["data"]["length"] = data["data"][
            "lifespan"
        ]  # len(data["data"]["frames"])

        start_time = data["data"]["tracklet_start"]
        end_time = data["data"]["tracklet_end"]

        if pre_treatment and end_time >= 96:
            continue

        roots.append(node_id) if graph.in_degree(node_id) == 0 else None

    if len(roots) > 1:
        raise Exception

    for node_id in graph.nodes:
        if pre_treatment and graph.nodes[node_id]["data"]["tracklet_end"] >= 96:
            continue

        if graph.out_degree(node_id) == 0:
            if node_id in roots:
                start_time = graph.nodes[node_id]["data"]["tracklet_start"]
                length = graph.nodes[node_id]["data"]["lifespan"]

                path_info = [
                    {
                        "node_id": node_id,
                        "index": 0,
                        "start_time": start_time,
                        "length": length,
                        "fate": graph.nodes[node_id]["data"]["fate"],
                    }
                ]
                division_profile_records.append(
                    {
                        "node_id": node_id,
                        "start": start_time,
                        "end": start_time + length - 1,
                        "first_length": length,
                        "dosage": dosage,
                        "generations": 0,
                        "path": path_info,
                    }
                )
            else:
                for root in roots:
                    if nx.shortest_path_length(graph, root, node_id):
                        path = nx.shortest_path(graph, root, node_id)
                        path_info = []

                        start = graph.nodes[path[0]]["data"]["tracklet_start"]
                        end = graph.nodes[path[-1]]["data"]["tracklet_end"]
                        first_length = graph.nodes[path[0]]["data"]["lifespan"]

                        for idx, step_node in enumerate(path):
                            length = graph.nodes[step_node]["data"]["lifespan"]
                            start_index = graph.nodes[step_node]["data"][
                                "tracklet_start"
                            ]
                            path_info.append(
                                {
                                    "node_id": step_node,
                                    "index": idx,
                                    "start_time": start_index,
                                    "length": length,
                                    "fate": graph.nodes[step_node]["data"]["fate"],
                                }
                            )
                        division_profile_records.append(
                            {
                                "node_id": node_id,
                                "start": start,
                                "end": end,
                                "first_length": first_length,
                                "dosage": dosage,
                                "generations": len(path) - 1,
                                "path": path_info,
                            }
                        )
                        break

    return division_profile_records


def analyse_subgraph(graph, dosage, lineage_id, pre_treatment):
    records = []
    time_resolved_records = []
    division_timing_records = []

    roots = []

    for node_id, data in graph.nodes(data=True):
        bboxes = [frame["cell"]["bounding_box"] for frame in data["data"]["frames"]]
        graph.nodes[node_id]["data"]["size_slope"] = compute_size_slope(bboxes)
        graph.nodes[node_id]["data"]["mean_area"] = compute_mean_area(bboxes)
        roots.append(node_id) if graph.in_degree(node_id) == 0 else None

    def walk_tree(node, root_record: Record, ancestor_record: Record, current_gen):
        if pre_treatment and graph.nodes[node]["data"]["frames"][-1]["frame_id"] >= 96:
            return

        node_record = Record(
            _id=node,
            fate=graph.nodes[node]["data"]["fate"],
            slope=graph.nodes[node]["data"]["size_slope"],
            mean_area=graph.nodes[node]["data"]["mean_area"],
            generation=current_gen,
            area_over_time=[
                (frame["cell"]["bounding_box"][2] - frame["cell"]["bounding_box"][0])
                * (frame["cell"]["bounding_box"][3] - frame["cell"]["bounding_box"][1])
                for frame in graph.nodes[node]["data"]["frames"][
                    : graph.nodes[node]["data"]["lifespan"]
                ]
            ],
            mean_intensity_over_time=[
                frame["cell"]["mean_intensity"]
                for frame in graph.nodes[node]["data"]["frames"][
                    : graph.nodes[node]["data"]["lifespan"]
                ]
            ],
        )

        for t, area in enumerate(node_record.get_area_over_time()):
            time_resolved_records.append(
                {
                    "lineage_id": lineage_id,
                    "generation": current_gen,
                    "time_index": t,
                    "area": area,
                    "dosage": dosage,
                }
            )

        if (
            graph.nodes[node]["data"]["fate"] == "mitosis"
            and node_record.get_fate_time() > 10
        ):
            division_timing_records.append(
                {
                    "fate_time": node_record.get_fate_time(),
                    "generation": current_gen,
                    "dosage": dosage,
                }
            )

        generation_offset = current_gen - root_record.get_generation()

        if generation_offset > 0:
            records.append(
                {
                    "root_fate": root_record.get_fate(),
                    "root_mean_area": root_record.get_mean_area(),
                    "root_slope": root_record.get_slope(),
                    "root_area_over_time": root_record.get_area_over_time(),
                    "root_mean_intensity_over_time": root_record.get_mean_intensity_over_time(),
                    "root_fate_time": root_record.get_fate_time(),
                    "ancestor_fate": ancestor_record.get_fate(),
                    "ancestor_id": ancestor_record.get_id(),
                    "ancestor_mean_area": ancestor_record.get_mean_area(),
                    "ancestor_slope": ancestor_record.get_slope(),
                    "ancestor_area_over_time": ancestor_record.get_area_over_time(),
                    "ancestor_mean_intensity_over_time": ancestor_record.get_mean_intensity_over_time(),
                    "ancestor_fate_time": ancestor_record.get_fate_time(),
                    "descendant_id": node,
                    "descendant_fate": node_record.get_fate(),
                    "descendant_mean_area": node_record.get_mean_area(),
                    "descendant_slope": node_record.get_slope(),
                    "descendant_area_over_time": node_record.get_area_over_time(),
                    "descendant_mean_intensity_over_time": node_record.get_mean_intensity_over_time(),
                    "descendant_fate_time": node_record.get_fate_time(),
                    "generation": generation_offset,
                    "dosage": dosage,
                    "lineage_id": lineage_id,
                }
            )
        for child in graph.successors(node):
            walk_tree(
                child,
                root_record,
                node_record,
                current_gen + 1,
            )

    for root in roots:
        start_time = graph.nodes[root]["data"]["frames"][0]["frame_id"]

        if pre_treatment and start_time >= 96:
            continue

        root_record = Record(
            _id=root,
            fate=graph.nodes[root]["data"]["fate"],
            generation=0,
            mean_area=graph.nodes[root]["data"]["mean_area"],
            slope=graph.nodes[root]["data"]["size_slope"],
            area_over_time=[
                (frame["cell"]["bounding_box"][2] - frame["cell"]["bounding_box"][0])
                * (frame["cell"]["bounding_box"][3] - frame["cell"]["bounding_box"][1])
                for frame in graph.nodes[root]["data"]["frames"]
            ],
            mean_intensity_over_time=[
                frame["cell"]["mean_intensity"]
                for frame in graph.nodes[root]["data"]["frames"]
            ],
        )

        if root_record.get_fate() == "mitosis" and root_record.get_fate_time() > 10:
            division_timing_records.append(
                {
                    "fate_time": root_record.get_fate_time(),
                    "dosage": dosage,
                    "generation": 0,
                }
            )

        walk_tree(
            root,
            root_record,
            ancestor_record=None,
            current_gen=0,
        )

    return (records, time_resolved_records, division_timing_records)


def filter_subgraphs(subgraphs, start_time=1):
    valid_subgraphs = []
    for subgraph in subgraphs:
        root_candidates = [
            node for node in subgraph.nodes if subgraph.in_degree(node) == 0
        ]
        if not root_candidates:
            continue

        root = root_candidates[0]
        beginning = subgraph.nodes[root]["data"]["frames"][0]["frame_id"]
        if beginning == start_time:
            valid_subgraphs.append(subgraph)

    return valid_subgraphs


def analyze_video(
    summary_path,
    video,
    dosage,
    lineage_id,
    start_time=1,
    visualize=False,
    lower_bound=1,
    _filter=True,
    pre_treatment=False,
):
    G = nx.DiGraph()
    try:
        with open(summary_path, "r") as f:
            summary = json.load(f)
    except FileNotFoundError:
        # print(f"Summary file not found for video {video}. Skipping.")
        return [], [], [], [], lineage_id

    for tracklet in summary["tracklets"]:
        add_tracklet_to_graph(G, tracklet, visualize)

    for tracklet in summary["tracklets"]:
        if tracklet["childs"] is not None:
            for child in tracklet["childs"]:
                G.add_edge(tracklet["tracklet_id"], child)

    components = list(nx.weakly_connected_components(G))

    connected_subgraphs = [G.subgraph(component).copy() for component in components]

    if _filter:
        valid_subgraphs = filter_subgraphs(connected_subgraphs, start_time=start_time)
    else:
        valid_subgraphs = connected_subgraphs

    division_profile_records = []

    for subgraph in valid_subgraphs:
        division_profile_records.extend(
            division_profile_analysis(subgraph, dosage, pre_treatment=pre_treatment)
        )

    connected_subgraphs = [
        G.subgraph(component).copy() for component in components if len(component) > 0
    ]

    if _filter:
        valid_subgraphs = filter_subgraphs(connected_subgraphs, start_time=start_time)
    else:
        valid_subgraphs = connected_subgraphs

    records = []
    all_time_resolved_records = []
    division_timing_records = []

    for connected_subgraph in valid_subgraphs:
        lineage_id += 1
        record, time_resolved_record, division_timing_record = analyse_subgraph(
            connected_subgraph,
            dosage,
            lineage_id,
            pre_treatment=pre_treatment,
        )

        records.extend(record)
        all_time_resolved_records.extend(time_resolved_record)
        division_timing_records.extend(division_timing_record)

    if visualize:
        filtered_graph = nx.DiGraph()
        for sg in valid_subgraphs:
            filtered_graph.update(sg)

            net = Network(height="2000px", width="100%", directed=True)
            net.from_nx(filtered_graph)

        net.show(f"graph_{video}.html", notebook=False)

    return (
        records,
        all_time_resolved_records,
        division_timing_records,
        division_profile_records,
        lineage_id,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="src/data/analysis/evaluation")
    parser.add_argument("--output_dir", default="src/data/analysis/results")
    parser.add_argument("--skip_intensity", action="store_true", default=False)
    parser.add_argument("--num_frames", type=int, default=234)

    args = parser.parse_args()
    data_dir = args.data_dir
    # data_dir = "src/final_analysis/evaluation"
    output_dir = args.output_dir
    num_frames = args.num_frames

    os.makedirs(output_dir, exist_ok=True)

    videos = [
        f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))
    ]
    videos.sort()

    if not args.skip_intensity:
        print("Extracting mean intensity of each cell...")
        dataloader = CocoVidDataLoader(
            data_dir="src/data/dataset/", ann_file="annotations.json", img_type="rgb"
        )
        mean = dataloader.mean
        std = dataloader.std

        for video in tqdm(videos):
            position = video.split("/")[-1]
            video_dataset = InferenceDataset(
                data_dir=os.path.join("src/data/dataset/", position),
                mean=mean,
                std=std,
            )

            try:
                with open(os.path.join(args.data_dir, position, "summary.json")) as f:
                    data = json.load(f)
            except FileNotFoundError:
                print(f"No summary file found for video {video}. Skipping.")
                continue

            tracks = [t for t in data["tracklets"]]

            for idx, (img, annotations) in tqdm(
                enumerate(video_dataset, start=1), leave=False
            ):
                size = img.shape[2]
                for track in tracks:
                    for frame in track["frames"]:
                        if frame["frame_id"] < idx:
                            continue
                        elif frame["frame_id"] == idx:
                            # get bounding box
                            bbox = frame["cell"]["bounding_box"]

                            # get image crop
                            crop = normalize_crop(img)

                            bounding_box = [
                                max(0, min(bbox[0], size)),
                                max(0, min(bbox[1], size)),
                                max(0, min(bbox[2], size)),
                                max(0, min(bbox[3], size)),
                            ]
                            crop = crop[
                                bounding_box[1] : bounding_box[3],
                                bounding_box[0] : bounding_box[2],
                                :,
                            ]

                            # get mean intensity per channel
                            mean_intensity = np.mean(crop, axis=(0, 1))

                            # save results
                            frame["cell"]["mean_intensity"] = mean_intensity.tolist()
                        elif frame["frame_id"] > idx:
                            break

            # save updated data
            with open(os.path.join(args.data_dir, position, "summary.json"), "w") as f:
                json.dump(data, f)

        print("Finished extracting mean intensity of each cell.")
    print("Starting analysis of videos...")

    record_results = []
    time_resolved_results = []
    division_timing_results = []
    division_profile_results = []

    lineage_id = 0

    for idx, video in tqdm(enumerate(videos)):
        if video in ["052", "060", "256", "265"]:
            continue
        summary_path = os.path.join(data_dir, video, "summary.json")

        if idx == 0:
            with open(summary_path) as f:
                data = json.load(f)

            start_time = data["tracklets"][0]["frames"][0]["cell"]["time_step"]

        # get dose of video
        dosage = get_dosage(video)
        if dosage == "Unknown":
            continue

        (
            record_result,
            time_resolved_result,
            division_timing_result,
            division_profile_result,
            lineage_id,
        ) = analyze_video(
            summary_path,
            video,
            dosage,
            lineage_id,
            start_time=start_time,
            visualize=False,
            _filter=True,
            pre_treatment=False,
        )
        record_results.extend(record_result)
        time_resolved_results.extend(time_resolved_result)
        division_timing_results.extend(division_timing_result)
        division_profile_results.extend(division_profile_result)

    record_results = pd.DataFrame(record_results)

    conditional_probability_results = pd.DataFrame(record_results)

    division_timing_results = pd.DataFrame(division_timing_results)

    division_profile_results = pd.DataFrame(division_profile_results)

    # plots
    if not division_timing_results.empty:
        plot_division_timing(division_timing_results, output_dir, num_frames)
    else:
        print("No division timing results to plot.")
    if not record_results.empty:
        fig2 = plot_sister_cell(record_results, None, num_frames, 10)
        fig1 = plot_ancestor_descendant(record_results, None)
        combine_figures([fig1, fig2], "correlation_combined", output_dir)
        plot_root_descendant(record_results, output_dir)
        plot_fate_probability(record_results, output_dir, filter=True)
    else:
        print("No record results to plot.")
    if not division_profile_results.empty:
        plot_division_profile(division_profile_results, output_dir, num_frames)
    else:
        print("No division profile results to plot.")
