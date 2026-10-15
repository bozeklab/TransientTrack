import os

import cv2
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
from scipy.stats import pearsonr

from utils.analysis_plots import (
    create_ancestor_descendant_plot,
    create_cell_cycle_corr_heatmap,
    create_division_boxplots,
    create_division_profile_plot,
    create_fate_probability_plot,
    create_root_descendant_plot,
    create_sister_cell_plot,
    create_time_resolved_plot,
)


def normalize_crop(crop):
    crop = crop.squeeze(0)
    return cv2.normalize(
        np.transpose(crop.numpy(), (1, 2, 0)), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)


def compute_size_slope(bboxes):
    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in bboxes]
    frames = np.arange(len(areas))

    if len(areas) < 2:
        return None

    slope, _ = np.polyfit(frames, areas, 1)
    return slope


def compute_mean_area(bboxes):
    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in bboxes]
    if not areas:
        return None
    return np.mean(areas)


def split_intensities(intensities):
    intensities_r, intensities_g, intensities_b = zip(*intensities)
    return np.array(intensities_r), np.array(intensities_g), np.array(intensities_b)


def plot_fate_probability(all_data, output_dir, filter=True, threshold=10):
    all_data["fate_transition"] = (
        all_data["ancestor_fate"] + " → " + all_data["descendant_fate"]
    )

    grouped_counts = (
        all_data.groupby(["dosage", "fate_transition", "generation"])
        .size()
        .reset_index(name="count")
    )

    if filter:
        total_counts = (
            grouped_counts.groupby(["dosage", "generation"])["count"]
            .sum()
            .reset_index(name="total_count")
        )
        filtered = pd.merge(grouped_counts, total_counts, on=["dosage", "generation"])
        filtered = filtered[filtered["total_count"] >= threshold]
        data = filtered
    else:
        data = grouped_counts

    grouped_probs = (
        data.groupby(["dosage", "generation"])
        .apply(lambda df: df.assign(prob=df["count"] / df["count"].sum()))
        .reset_index(drop=True)
    )

    generation_distribution = (
        all_data.groupby(["dosage", "generation"]).size().reset_index(name="count")
    )
    generation_distribution["generation"] = pd.to_numeric(
        generation_distribution["generation"]
    )

    if filter:
        max_generation = grouped_probs["generation"].max()
        generation_distribution = generation_distribution[
            generation_distribution["generation"] <= max_generation
        ]

    create_fate_probability_plot(grouped_probs, generation_distribution, output_dir)


def plot_root_descendant(all_data, output_dir, filter=True):
    results = []
    cell_cycle_corr = []

    for _, group in all_data.groupby(["lineage_id", "dosage"]):
        root_area_raw = group.iloc[0]["root_area_over_time"]
        root_intensities_r_raw, root_intensities_g_raw, root_intensities_b_raw = (
            split_intensities(group.iloc[0]["root_mean_intensity_over_time"])
        )

        for _, row in group.iterrows():
            if filter and row["generation"] > 5:
                continue
            desc_area = row["descendant_area_over_time"]
            desc_intensities_r, desc_intensities_g, desc_intensities_b = (
                split_intensities(row["descendant_mean_intensity_over_time"])
            )

            if (
                not np.isnan(row["descendant_fate_time"])
                and row["root_fate"] == row["descendant_fate"]
            ):
                cell_cycle_corr.append(
                    {
                        "lineage_id": row["lineage_id"],
                        "dosage": row["dosage"],
                        "generation": row["generation"],
                        "root_cell_cycle_time": row["root_fate_time"],
                        "descendant_cell_cycle_time": row["descendant_fate_time"],
                    }
                )

            desc_intensities_r, desc_intensities_g, desc_intensities_b = (
                desc_intensities_r[~np.isnan(desc_intensities_r)],
                desc_intensities_g[~np.isnan(desc_intensities_g)],
                desc_intensities_b[~np.isnan(desc_intensities_b)],
            )

            min_area_len = min(len(root_area_raw), len(desc_area))
            root_area, desc_area = (
                root_area_raw[:min_area_len],
                desc_area[:min_area_len],
            )

            root_intensities_r_raw, root_intensities_g_raw, root_intensities_b_raw = (
                root_intensities_r_raw[~np.isnan(root_intensities_r_raw)],
                root_intensities_g_raw[~np.isnan(root_intensities_g_raw)],
                root_intensities_b_raw[~np.isnan(root_intensities_b_raw)],
            )

            min_int_len = min(len(root_intensities_r_raw), len(desc_intensities_r))

            root_intensities_r, desc_intensities_r = (
                root_intensities_r_raw[:min_int_len],
                desc_intensities_r[:min_int_len],
            )
            root_intensities_g, desc_intensities_g = (
                root_intensities_g_raw[:min_int_len],
                desc_intensities_g[:min_int_len],
            )
            root_intensities_b, desc_intensities_b = (
                root_intensities_b_raw[:min_int_len],
                desc_intensities_b[:min_int_len],
            )

            if min_area_len >= 3:
                area_correlation_pearson, area_correlation_pearson_p = pearsonr(
                    root_area, desc_area
                )
            else:
                area_correlation_pearson, area_correlation_pearson_p = None, None
            if min_int_len >= 3:
                intensity_correlation_pearson_r, intensity_correlation_r_p = pearsonr(
                    root_intensities_r, desc_intensities_r
                )
                intensity_correlation_pearson_g, intensity_correlation_g_p = pearsonr(
                    root_intensities_g, desc_intensities_g
                )
                intensity_correlation_pearson_b, intensity_correlation_b_p = pearsonr(
                    root_intensities_b, desc_intensities_b
                )
            else:
                (
                    intensity_correlation_pearson_r,
                    intensity_correlation_pearson_g,
                    intensity_correlation_pearson_b,
                    intensity_correlation_r_p,
                    intensity_correlation_g_p,
                    intensity_correlation_b_p,
                ) = (None, None, None, None, None, None)

            results.append(
                {
                    "lineage_id": row["lineage_id"],
                    "dosage": row["dosage"],
                    "generation": row["generation"],
                    "mean_area_diff": abs(
                        row["descendant_mean_area"] - row["root_mean_area"]
                    ),
                    "slope_diff": abs(row["descendant_slope"] - row["root_slope"]),
                    "area_correlation_pearson": area_correlation_pearson,
                    "area_correlation_pearson_p": area_correlation_pearson_p,
                    "intensity_correlation_pearson_r": intensity_correlation_pearson_r,
                    "intensity_correlation_pearson_r_p": intensity_correlation_r_p,
                    "intensity_correlation_pearson_g": intensity_correlation_pearson_g,
                    "intensity_correlation_pearson_g_p": intensity_correlation_g_p,
                    "intensity_correlation_pearson_b": intensity_correlation_pearson_b,
                    "intensity_correlation_pearson_b_p": intensity_correlation_b_p,
                    "fates_equal": row["descendant_fate"] == row["root_fate"],
                }
            )
    results_df = pd.DataFrame(results)

    cell_cycle_corr = pd.DataFrame(cell_cycle_corr)

    cell_cycle_results = (
        cell_cycle_corr.groupby(["dosage", "generation"])
        .apply(
            lambda g: g["root_cell_cycle_time"].corr(
                g["descendant_cell_cycle_time"], method="pearson"
            )
        )
        .reset_index(name="pearson_corr")
    )

    create_cell_cycle_corr_heatmap(cell_cycle_results, "root_desc", output_dir)

    create_root_descendant_plot(results_df, output_dir)


def plot_ancestor_descendant(all_data, output_dir, filter=True):
    results = []
    cell_cycle_corr = []

    for _, group in all_data.groupby(["lineage_id", "dosage"]):
        for _, row in group.iterrows():
            if filter and row["generation"] > 5:
                continue
            ancestor_area = row["ancestor_area_over_time"]
            desc_area = row["descendant_area_over_time"]

            if (
                not np.isnan(row["descendant_fate_time"])
                # and row["ancestor_fate"] == row["descendant_fate"]
            ):
                cell_cycle_corr.append(
                    {
                        "lineage_id": row["lineage_id"],
                        "dosage": row["dosage"],
                        "generation": row["generation"],
                        "ancestor_cell_cycle_time": row["ancestor_fate_time"],
                        "descendant_cell_cycle_time": row["descendant_fate_time"],
                    }
                )

            min_area_len = min(len(ancestor_area), len(desc_area))
            ancestor_area, desc_area = (
                ancestor_area[:min_area_len],
                desc_area[:min_area_len],
            )

            (
                ancestor_intensities_r_raw,
                ancestor_intensities_g_raw,
                ancestor_intensities_b_raw,
            ) = split_intensities(row["ancestor_mean_intensity_over_time"])
            desc_intensities_r_raw, desc_intensities_g_raw, desc_intensities_b_raw = (
                split_intensities(row["descendant_mean_intensity_over_time"])
            )

            desc_intensities_r_raw, desc_intensities_g_raw, desc_intensities_b_raw = (
                desc_intensities_r_raw[~np.isnan(desc_intensities_r_raw)],
                desc_intensities_g_raw[~np.isnan(desc_intensities_g_raw)],
                desc_intensities_b_raw[~np.isnan(desc_intensities_b_raw)],
            )

            min_int_len = min(
                len(ancestor_intensities_r_raw), len(desc_intensities_r_raw)
            )

            # min_len = min(len(ancestor_area), len(desc_area))
            ancestor_intensities_r, desc_intensities_r = (
                ancestor_intensities_r_raw[:min_int_len],
                desc_intensities_r_raw[:min_int_len],
            )
            ancestor_intensities_g, desc_intensities_g = (
                ancestor_intensities_g_raw[:min_int_len],
                desc_intensities_g_raw[:min_int_len],
            )
            ancestor_intensities_b, desc_intensities_b = (
                ancestor_intensities_b_raw[:min_int_len],
                desc_intensities_b_raw[:min_int_len],
            )

            if min_area_len >= 3:
                area_correlation_pearson, area_correlation_pearson_p = pearsonr(
                    ancestor_area, desc_area
                )
            else:
                area_correlation_pearson, area_correlation_pearson_p = None, None

            if min_int_len >= 3:
                intensity_correlation_pearson_r, intensity_correlation_r_p = pearsonr(
                    ancestor_intensities_r, desc_intensities_r
                )
                intensity_correlation_pearson_g, intensity_correlation_g_p = pearsonr(
                    ancestor_intensities_g, desc_intensities_g
                )
                intensity_correlation_pearson_b, intensity_correlation_b_p = pearsonr(
                    ancestor_intensities_b, desc_intensities_b
                )
            else:
                (
                    intensity_correlation_pearson_r,
                    intensity_correlation_pearson_g,
                    intensity_correlation_pearson_b,
                    intensity_correlation_r_p,
                    intensity_correlation_g_p,
                    intensity_correlation_b_p,
                ) = (None, None, None, None, None, None)

            results.append(
                {
                    "lineage_id": row["lineage_id"],
                    "dosage": row["dosage"],
                    "generation": row["generation"],
                    "mean_area_diff": abs(
                        row["descendant_mean_area"] - row["ancestor_mean_area"]
                    ),
                    "slope_diff": abs(row["descendant_slope"] - row["ancestor_slope"]),
                    "area_correlation_pearson": area_correlation_pearson,
                    "area_correlation_pearson_p": area_correlation_pearson_p,
                    "intensity_correlation_pearson_r": intensity_correlation_pearson_r,
                    "intensity_correlation_pearson_r_p": intensity_correlation_r_p,
                    "intensity_correlation_pearson_g": intensity_correlation_pearson_g,
                    "intensity_correlation_pearson_g_p": intensity_correlation_g_p,
                    "intensity_correlation_pearson_b": intensity_correlation_pearson_b,
                    "intensity_correlation_pearson_b_p": intensity_correlation_b_p,
                    "fates_equal": row["descendant_fate"] == row["ancestor_fate"],
                }
            )
    results_df = pd.DataFrame(results)

    results_df = results_df[results_df["generation"] > 1]
    # results_df = results_df[]

    cell_cycle_corr = pd.DataFrame(cell_cycle_corr)

    # cell_cycle_corr = cell_cycle_corr[
    #     (cell_cycle_corr["ancestor_cell_cycle_time"].between(10, 100))
    #     & (cell_cycle_corr["descendant_cell_cycle_time"].between(10, 100))
    # ]
    cell_cycle_corr = cell_cycle_corr[cell_cycle_corr["generation"] > 1]

    cell_cycle_corr = cell_cycle_corr[cell_cycle_corr["ancestor_cell_cycle_time"] > 10]
    # cell_cycle_corr = cell_cycle_corr[cell_cycle_corr["ancestor_cell_cycle_time"] < 100]
    cell_cycle_corr = cell_cycle_corr[
        cell_cycle_corr["descendant_cell_cycle_time"] > 10
    ]
    # cell_cycle_corr = cell_cycle_corr[
    #     cell_cycle_corr["descendant_cell_cycle_time"] < 100
    # ]

    cell_cycle_results = (
        cell_cycle_corr.groupby(["dosage", "generation"])
        .apply(
            lambda g: g["ancestor_cell_cycle_time"].corr(
                g["descendant_cell_cycle_time"], method="pearson"
            )
        )
        .reset_index(name="pearson_corr")
    )

    # create_cell_cycle_corr_heatmap(cell_cycle_results, "anc_desc", output_dir)

    if output_dir:
        create_ancestor_descendant_plot(results_df, output_dir)
    else:
        return create_ancestor_descendant_plot(results_df, None)


def remove_nan_values(lst: list):
    lst = [x for x in lst if not np.isnan(x)]
    return lst


def plot_sister_cell(all_data, output_dir, num_frames, min_pairs=10, filter=True):
    sister_results = []
    for _, group in all_data.groupby(["lineage_id", "generation"]):
        if len(group) < 2:
            continue
        row_a, row_b = group.iloc[0], group.iloc[1]

        if filter and row_a["generation"] > 5:
            continue

        areas_a = row_a["descendant_area_over_time"]
        areas_b = row_b["descendant_area_over_time"]

        min_area_len = min(len(areas_a), len(areas_b))
        areas_a, areas_b = areas_a[:min_area_len], areas_b[:min_area_len]

        intensities_a_r, intensities_a_g, intensities_a_b = split_intensities(
            row_a["descendant_mean_intensity_over_time"]
        )
        intensities_b_r, intensities_b_g, intensities_b_b = split_intensities(
            row_b["descendant_mean_intensity_over_time"]
        )

        intensities_a_r = intensities_a_r[~np.isnan(intensities_a_r)]
        intensities_a_g = intensities_a_g[~np.isnan(intensities_a_g)]
        intensities_b_g = intensities_b_g[~np.isnan(intensities_b_g)]
        intensities_a_b = intensities_a_b[~np.isnan(intensities_a_b)]
        intensities_b_b = intensities_b_b[~np.isnan(intensities_b_b)]

        min_int_len = min(len(intensities_a_r), len(intensities_b_r))

        intensities_a_r, intensities_b_r = (
            intensities_a_r[:min_int_len],
            intensities_b_r[:min_int_len],
        )
        intensities_a_g, intensities_b_g = (
            intensities_a_g[:min_int_len],
            intensities_b_g[:min_int_len],
        )
        intensities_a_b, intensities_b_b = (
            intensities_a_b[:min_int_len],
            intensities_b_b[:min_int_len],
        )

        if min_area_len >= 3:
            area_correlation_pearson, area_correlation_pearson_p = pearsonr(
                areas_a, areas_b
            )
        else:
            area_correlation_pearson, area_correlation_pearson_p = None, None
        if min_int_len >= 3:
            intensity_correlation_pearson_r, intensity_correlation_r_p = pearsonr(
                intensities_a_r, intensities_b_r
            )
            intensity_correlation_pearson_g, intensity_correlation_g_p = pearsonr(
                intensities_a_g, intensities_b_g
            )
            intensity_correlation_pearson_b, intensity_correlation_b_p = pearsonr(
                intensities_a_b, intensities_b_b
            )
        else:
            (
                intensity_correlation_pearson_r,
                intensity_correlation_pearson_g,
                intensity_correlation_pearson_b,
                intensity_correlation_r_p,
                intensity_correlation_g_p,
                intensity_correlation_b_p,
            ) = (None, None, None, None, None, None)

        # TODO:
        fate_delta = None

        if row_a["descendant_fate"] != "alive" and row_b["descendant_fate"] != "alive":
            fate_delta = abs(
                row_a["descendant_fate_time"] - row_b["descendant_fate_time"]
            )

        sister_results.append(
            {
                "generation": row_a["generation"],
                "lineage_id": row_a["lineage_id"],
                "dosage": row_a["dosage"],
                "alive_fates_equal": row_a["descendant_fate"]
                == row_b["descendant_fate"]
                == "alive",
                "div_fates_equal": row_a["descendant_fate"]
                == row_b["descendant_fate"]
                == "mitosis",
                "death_fates_equal": row_a["descendant_fate"]
                == row_b["descendant_fate"]
                == "apoptosis",
                "fate_delta": fate_delta,
                "mean_area_diff": abs(
                    row_a["descendant_mean_area"] - row_b["descendant_mean_area"]
                ),
                "slope_diff": abs(
                    row_a["descendant_slope"] - row_b["descendant_slope"]
                ),
                "area_correlation_pearson": area_correlation_pearson,
                "area_correlation_pearson_p": area_correlation_pearson_p,
                "intensity_correlation_pearson_r": intensity_correlation_pearson_r,
                "intensity_correlation_pearson_r_p": intensity_correlation_r_p,
                "intensity_correlation_pearson_g": intensity_correlation_pearson_g,
                "intensity_correlation_pearson_g_p": intensity_correlation_g_p,
                "intensity_correlation_pearson_b": intensity_correlation_pearson_b,
                "intensity_correlation_pearson_b_p": intensity_correlation_b_p,
            }
        )

    sister_df = pd.DataFrame(sister_results)

    fate_sync_stats = (
        sister_df.groupby(["generation", "dosage"])
        .agg(
            total_pairs=("div_fates_equal", "count"),
            matching_alive_fates=("alive_fates_equal", "sum"),
            matching_div_fates=("div_fates_equal", "sum"),
            matching_death_fates=("death_fates_equal", "sum"),
        )
        .reset_index()
    )

    fate_sync_stats = fate_sync_stats[fate_sync_stats["total_pairs"] >= min_pairs]

    fate_sync_stats["alive_fate_sync_proportion"] = (
        fate_sync_stats["matching_alive_fates"] / fate_sync_stats["total_pairs"]
    )

    fate_sync_stats["div_fate_sync_proportion"] = (
        fate_sync_stats["matching_div_fates"] / fate_sync_stats["total_pairs"]
    )
    fate_sync_stats["death_fate_sync_proportion"] = (
        fate_sync_stats["matching_death_fates"] / fate_sync_stats["total_pairs"]
    )

    fate_sync_stats = pd.melt(
        fate_sync_stats,
        id_vars=["generation", "dosage"],
        value_vars=[
            "alive_fate_sync_proportion",
            "div_fate_sync_proportion",
            "death_fate_sync_proportion",
        ],
        var_name="fate_type",
        value_name="sync_proportion",
    )

    if output_dir:
        create_sister_cell_plot(sister_df, fate_sync_stats, output_dir)
    else:
        return create_sister_cell_plot(sister_df, fate_sync_stats, output_dir)

    # create_histograms(
    #     sister_df,
    #     prop="fate_delta",
    #     title="Sister Cell Fate Delta Histogram",
    #     xaxis_title="Fate Delta (0.5hr)",
    #     output_dir=output_dir,
    #     show_mean=False,
    #     file_name="sister_cell_fate_delta_histogram",
    # )


def plot_division_timing(all_data, output_dir, num_frames):

    # drop data with generation >= 5
    data = all_data[all_data["generation"] < 5]

    # discard generation 0
    data = data[data["generation"] > 0]

    data = data[data["fate_time"] < 100]

    # create_histograms(
    #     data,
    #     prop="fate_time",
    #     title="Division Timing Histogram by Dosage",
    #     xaxis_title="Division Time (0.5hr)",
    #     output_dir=output_dir,
    #     file_name="division_timing_histogram",
    # )

    create_division_boxplots(
        data,
        prop="fate_time",
        title="Time Until Division Histogram by Dosage",
        output_dir=output_dir,
        file_name="division_timing_histogram",
    )

    # for dosage, group in data.groupby("dosage"):
    #     create_division_boxplots(
    #         group,
    #         prop="fate_time",
    #         title=f"Division Timing Boxplots for Dosage {dosage}",
    #         output_dir=output_dir,
    #         file_name=f"division_timing_boxplots_dosage_{dosage}",
    #     )

    # create_division_boxplots(
    #     all_data,
    #     prop="fate_time",
    #     title="Division Timing Boxplots",
    #     xaxis_title="Division Time (0.5hr)",
    #     output_dir=output_dir,
    #     file_name="division_timing_boxplots",
    # )


def plot_time_resolved_results(all_data, output_dir):
    create_time_resolved_plot(all_data, output_dir)


def plot_division_profile(all_data, output_dir, num_frames):
    rows = []

    for path_index, row in all_data.iterrows():
        dosage = row["dosage"]
        path = row["path"]
        generations = row["generations"]

        for step in path:
            # if step.get("fate") != "mitosis":
            #     continue
            rows.append(
                {
                    "path_index": path_index,
                    "node_id": step.get("node_id"),
                    "index": step.get("index"),
                    "start_time": step.get("start_time"),
                    "length": step.get("length"),
                    "generation": path.index(step),
                    "generations": generations,
                    "fate": step.get("fate"),
                    "dosage": dosage,
                }
            )
    data = pd.DataFrame(rows)

    sorted_path_indices = []

    for gen_count, group in data.groupby("generations"):
        max_index = group["index"].max()
        full_index_range = list(range(max_index + 1))

        pivot = group.pivot(index="path_index", columns="index", values="length")
        for col in full_index_range:
            if col not in pivot.columns:
                pivot[col] = np.inf

        pivot_sorted = pivot[full_index_range].sort_values(list(full_index_range))

        sorted_path_indices.extend(pivot_sorted.index.tolist())

    index_map = {
        old_index: new_index for new_index, old_index in enumerate(sorted_path_indices)
    }

    data["path_index_new"] = data["path_index"].map(index_map)

    extensions = []

    bad_rows = []

    for i, row in data.iterrows():
        # delete rows that do not go until the end of the sequence (num_frames)
        if row["start_time"] + row["length"] < num_frames:
            # data.loc[i, "length"] = num_frames - row["start_time"]
            if row["fate"] == "apoptosis":
                extensions.append(
                    {
                        "path_index_new": row["path_index_new"],
                        "start_time": row["start_time"] + row["length"],
                        "length": num_frames - (row["start_time"] + row["length"]),
                        "generation": row["generation"] + 1,
                        "fate": "apoptosis-extension",
                        "dosage": row["dosage"],
                    }
                )

    last_segments = (
        data.sort_values("index").groupby("path_index_new").last().reset_index()
    )

    for i, row in last_segments.iterrows():
        if row["start_time"] + row["length"] < num_frames and row["fate"] == "alive":
            bad_rows.append(row["path_index_new"])

    # Remove bad rows from data
    data = data[~data["path_index_new"].isin(bad_rows)].copy()

    extensions_df = pd.DataFrame(extensions)

    extensions_df["index"] = None
    extensions_df["node_id"] = None
    extensions_df["index"] = None
    extensions_df["path_index"] = None

    data = pd.concat([data, extensions_df], ignore_index=True)

    create_division_profile_plot(data, output_dir, num_frames, subsampling=300)


def combine_figures(figures, file_name, output_dir):
    num_figures = len(figures)

    combined_fig = make_subplots(
        rows=num_figures,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["Ancestor-Descendant", "Sister Cells"],
        vertical_spacing=0.1,
    )

    for i, fig in enumerate(figures):
        for trace in fig.data:
            trace.showlegend = i == 1 and trace.showlegend
            combined_fig.add_trace(trace, row=i + 1, col=1)

        if "shapes" in fig.layout:
            for shape in fig.layout.shapes:
                shape.showlegend = i == 1 and shape.showlegend
                if shape.layer == "below":
                    shape.line.width = 0
                combined_fig.add_shape(shape, row=i + 1, col=1)

        if "annotations" in fig.layout:
            for annotation in fig.layout.annotations:
                combined_fig.add_annotation(annotation, row=i + 1, col=1)

        if "legend" in fig.layout and i == 1:
            combined_fig.update_layout(legend=fig.layout.legend)

        if "legend2" in fig.layout and i == 1:
            combined_fig.update_layout(legend2=fig.layout.legend2)

        combined_fig.update_yaxes(title=fig.layout.yaxis.title, col=1)
        combined_fig.update_xaxes(
            title=fig.layout.xaxis.title if i == 1 else None,
            row=i + 1,
            col=1,
        )

    combined_fig.update_layout(
        height=1200,  # 1200
        width=1200,
        margin=dict(r=10, t=50, b=100),
        showlegend=True,
        template="plotly_white",
        legend=dict(
            orientation="h",
            # yanchor="bottom",
            # xanchor="left",
            x=-0.05,
            y=-0.09,
            # itemwidth=25,
            itemsizing="constant",
            indentation=15,
            font=dict(size=34, family="Times New Roman", color="black"),
        ),
        legend2=dict(
            orientation="h",
            x=-0.05,
            y=-0.14,
            itemsizing="constant",
            indentation=15,
            font=dict(size=34, family="Times New Roman", color="black"),
        ),
    )

    combined_fig.update_xaxes(
        # range=[x_min, x_max],
        tickmode="array",
        tick0=1,
        dtick=1,
        range=[0.5, 5.5],
        matches="x",
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        # showline=False,
        zeroline=False,
        title_font=dict(size=34, family="Times New Roman", color="black"),
        tickfont=dict(size=28, family="Times New Roman", color="black"),
    )

    for annotation in combined_fig.layout.annotations:
        annotation["font"] = dict(
            size=26, color="black", family="Times New Roman", weight="bold"
        )

    combined_fig.add_annotation(
        x=-0.05,
        y=1.058,
        text="(a)",
        showarrow=False,
        font=dict(size=34, family="Times New Roman", color="black", weight="bold"),
        xref="paper",
        yref="paper",
        align="center",
    )

    combined_fig.add_annotation(
        x=-0.05,
        y=0.485,
        text="(b)",
        showarrow=False,
        font=dict(size=34, family="Times New Roman", color="black", weight="bold"),
        xref="paper",
        yref="paper",
        align="center",
    )

    combined_fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        zeroline=False,
        tick0=-1,
        dtick=0.25,
        title_font=dict(size=34, family="Times New Roman", color="black"),
        tickfont=dict(size=28, family="Times New Roman", color="black"),
    )

    combined_fig.layout.annotations[0].font.size = 38
    combined_fig.layout.annotations[1].font.size = 38
    # print(combined_fig.layout)
    # combined_fig

    combined_fig.write_image(os.path.join(output_dir, f"{file_name}.eps"))


class Record:
    def __init__(
        self,
        _id=None,
        generation=None,
        fate=None,
        mean_area=None,
        slope=None,
        lineage_id=None,
        area_over_time=None,
        mean_intensity_over_time=None,
    ):
        self.id = _id
        self.generation = generation
        self.fate = fate
        self.mean_area = mean_area
        self.slope = slope
        self.lineage_id = lineage_id
        self.area_over_time = area_over_time
        self.mean_intensity_over_time = mean_intensity_over_time

        self.fate_time = (
            len(area_over_time)
            if self.fate == "apoptosis" or self.fate == "mitosis"
            else None
        )

    def get_id(self):
        return self.id

    def get_generation(self):
        return self.generation

    def get_fate(self):
        return self.fate

    def get_mean_area(self):
        return self.mean_area

    def get_slope(self):
        return self.slope

    def get_lineage_id(self):
        return self.lineage_id

    def get_area_over_time(self):
        return self.area_over_time

    def get_mean_intensity_over_time(self):
        return self.mean_intensity_over_time

    def get_fate_time(self):
        return self.fate_time
