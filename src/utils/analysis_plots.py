import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

# from plotly_formatter import PlotlyFormatter
from scipy.stats import gaussian_kde

COLOR_PALETTE = {
    "Low": "rgb(0, 128, 0)",
    "Medium": "rgb(255, 165, 0)",
    "High": "rgb(255, 0, 0)",
    "Control": "rgb(0, 0, 255)",
}

REGIONS = [
    (
        0.0,
        0.199,
        "rgb(210, 240, 210)",
        "very weak",
    ),
    (0.2, 0.399, "rgb(153, 216, 153)", "weak"),
    (0.4, 0.599, "rgb(102, 194, 102)", "moderate"),
    (0.6, 0.799, "rgb(35, 139, 69)", "strong"),
    (0.8, 1.0, "rgb(0, 90, 50)", "very strong"),
]

TITLE_FONT = dict(size=22, family="Times New Roman", color="black")
TICK_FONT = dict(size=16, family="Times New Roman", color="black")


# STANDARDIZER = PlotlyFormatter(
#     height=1600,
#     title_font=dict(size=22, family="Times New Roman", color="black"),
#     tick_font=dict(size=16, family="Times New Roman", color="black"),
#     legend_font=dict(size=16, family="Times New Roman", color="black"),
# )


def create_fate_probability_plot(grouped_probs, generation_distribution, output_dir):
    titles = [k for k in COLOR_PALETTE.keys()] + ["Support per generation"] * len(
        COLOR_PALETTE.keys()
    )

    fig = make_subplots(
        rows=2,
        cols=4,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.1,
        horizontal_spacing=0.01,
        row_heights=[0.6, 0.4],
        subplot_titles=titles,
    )

    transition_colors = {
        "mitosis → mitosis": "green",
        "mitosis → apoptosis": "red",
        "mitosis → alive": "gray",
    }

    transition_symbols = {
        "mitosis → mitosis": "circle",
        "mitosis → apoptosis": "x",
        "mitosis → alive": "square",
    }

    for (dosage, fate_transition), df in grouped_probs.groupby(
        ["dosage", "fate_transition"]
    ):
        fig.add_trace(
            go.Scatter(
                x=df["generation"],
                y=df["prob"],
                mode="lines+markers",
                name=f"{dosage} | {fate_transition}",
                line=dict(color=transition_colors.get(fate_transition, "gray")),
                marker=dict(
                    symbol=transition_symbols.get(fate_transition, "circle"),
                    size=10,
                ),
                showlegend=False,
            ),
            row=1,
            col=list(COLOR_PALETTE.keys()).index(dosage) + 1,
        )

    for dosage, df in generation_distribution.groupby("dosage"):
        fig.add_trace(
            go.Scatter(
                x=df["generation"],
                y=df["count"],
                mode="lines+markers",
                name=f"{dosage} (count)",
                line=dict(color="blue"),
                showlegend=False,
            ),
            row=2,
            col=list(COLOR_PALETTE.keys()).index(dosage) + 1,
        )

    # dummy traces
    for transition, color in transition_colors.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=color),
                name=f"{transition}",
                legendgroup="transition",
                showlegend=True,
            )
        )

    fig.update_yaxes(title_text="P(cell fate)", row=1, col=1)
    fig.update_yaxes(title_text="Number of transitions", row=2, col=1)
    fig.update_yaxes(row=2, type="log")

    fig.update_xaxes(title_text="Generation", row=2)

    fig.update_layout(
        title_text="Fate Probability Across Generations Per Dosage",
        title_x=0.5,
        legend_title="Fate Transition",
        hovermode="x unified",
    )

    # standardizer = PlotlyFormatter(
    #     height=1200,
    #     axis_title_ratio=1.35,
    #     tick_font_ratio=1.05,
    #     legend_font_ratio=1.25,
    #     subtitle_ratio=1.35,
    # )

    # standardizer.standardize(fig)

    output_file = os.path.join(output_dir, "fate_probability.png")
    # output_html = os.path.join(output_dir, "fate_probability.html")

    fig.write_image(output_file)
    # fig.write_html(output_html, include_plotlyjs=True)


def create_root_descendant_plot(data, output_dir):
    combined_data = data.copy()
    # map "High", "Medium", "Low" to "Cisplatin"
    combined_data["dosage"] = combined_data["dosage"].map(
        {
            "High": "Cisplatin",
            "Medium": "Cisplatin",
            "Low": "Cisplatin",
            "Control": "Control",
        }
    )
    create_plotly_boxes(
        combined_data,
        title="Root-Descendant Cell Size Correlation Across Generations",
        y_data="area_correlation_pearson",
        xaxis_title="Generation",
        yaxis_title="Pearson Correlation",
        output_dir=output_dir,
        file_name="root_descendant_size_correlation_pearson",
    )


def create_ancestor_descendant_plot(data, output_dir):
    combined_data = data.copy()
    # map "High", "Medium", "Low" to "Cisplatin"
    combined_data["dosage"] = combined_data["dosage"].map(
        {
            "High": "Cisplatin",
            "Medium": "Cisplatin",
            "Low": "Cisplatin",
            "Control": "Control",
        }
    )

    if output_dir:
        create_plotly_boxes(
            combined_data,
            title="Ancestor-Descendant Cell Size Correlation Across Generations",
            y_data="area_correlation_pearson",
            xaxis_title="Generation",
            yaxis_title="Pearson Correlation",
            output_dir=output_dir,
            file_name="ancestor_descendant_size_correlation_pearson",
        )
    else:
        return create_plotly_boxes(
            combined_data,
            title="Ancestor-Descendant Cell Size Correlation Across Generations",
            y_data="area_correlation_pearson",
            xaxis_title="Generation",
            yaxis_title="Pearson Correlation",
            output_dir=None,
            file_name="ancestor_descendant_size_correlation_pearson",
        )


def create_sister_cell_plot(data, fate_sync_stats, output_dir):
    combined_data = data.copy()

    # map "High", "Medium", "Low" to "Cisplatin"
    combined_data["dosage"] = combined_data["dosage"].map(
        {
            "High": "Cisplatin",
            "Medium": "Cisplatin",
            "Low": "Cisplatin",
            "Control": "Control",
        }
    )

    if output_dir:
        create_plotly_boxes(
            combined_data,
            title="Sister Cell Area Correlation Across Generations",
            y_data="area_correlation_pearson",
            xaxis_title="Generation",
            yaxis_title="Pearson Correlation",
            output_dir=output_dir,
            file_name="sister_cell_area_correlation_pearson_combined",
        )
    else:
        return create_plotly_boxes(
            combined_data,
            title="Sister Cell Area Correlation Across Generations",
            y_data="area_correlation_pearson",
            xaxis_title="Generation",
            yaxis_title="Pearson Correlation",
            output_dir=None,
            file_name="sister_cell_area_correlation_pearson_combined",
        )


def triangulate_histogram(x, y, z):
    if len(x) != len(y) != len(z):
        raise ValueError("The  lists x, y, z, must have the same length")
    n = len(x)
    if n % 2:
        raise ValueError("The length of lists x, y, z must be an even number")
    pts3d = np.vstack((x, y, z)).T
    pts3dp = np.array([[x[2 * k + 1], y[2 * k + 1], 0] for k in range(1, n // 2 - 1)])
    pts3d = np.vstack((pts3d, pts3dp))
    # triangulate the histogram bars:
    tri = [[0, 1, 2], [0, 2, n]]
    for k, i in zip(list(range(n, n - 3 + n // 2)), list(range(3, n - 4, 2))):
        tri.extend([[k, i, i + 1], [k, i + 1, k + 1]])
    tri.extend([[n - 3 + n // 2, n - 3, n - 2], [n - 3 + n // 2, n - 2, n - 1]])
    return pts3d, np.array(tri)


def create_division_boxplots(
    data,
    prop,
    title,
    output_dir,
    file_name,
):

    create_plotly_boxplot(
        data,
        title=title,
        y_data=prop,
        xaxis_title="Generation",
        yaxis_title="Time Until Division (0.5hr)",
        boxpoints="suspectedoutliers",
        boxmean=True,
        correlation=False,
        output_dir=output_dir,
        file_name=file_name,
    )


def create_histograms(
    data, prop, title, xaxis_title, output_dir, file_name, show_mean=True, only_kde=True
):
    DENSITY = True
    DENSITY_TYPE = "density"

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=1,
        vertical_spacing=0.1,
        subplot_titles=[k for k in data["dosage"].unique()],
    )
    hist_ymax = 0

    for idx, dosage in enumerate(data["dosage"].unique()):
        filtered_data = data[data["dosage"] == dosage]
        filtered_data = filtered_data[filtered_data["generation"] < 5]
        filtered_data = filtered_data[prop].dropna()

        bin_counts, bin_edges = np.histogram(filtered_data, bins=100, density=DENSITY)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        kde = gaussian_kde(filtered_data, bw_method=0.1)
        x_smooth = np.linspace(filtered_data.min(), filtered_data.max(), 100)

        if DENSITY:
            y_smooth = kde(x_smooth)
        else:
            y_smooth = kde(x_smooth) * len(filtered_data) * (x_smooth[1] - x_smooth[0])

        curr_hist_ymax = bin_counts.max()
        hist_ymax = max(hist_ymax, curr_hist_ymax, y_smooth.max()) * 1.01

        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=bin_counts,
                width=(bin_edges[1] - bin_edges[0]),
                marker_color=COLOR_PALETTE.get(dosage, "gray"),
                showlegend=False,
            ),
            row=idx + 1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=x_smooth,
                y=y_smooth,
                mode="lines",
                line=dict(color="black", width=6),
                showlegend=False,
            ),
            row=idx + 1,
            col=1,
        )

        if show_mean:
            mean_val = filtered_data.mean()
            std_val = filtered_data.std()

            # color = black as rgba
            fig.add_shape(
                type="line",
                x0=mean_val,
                y0=0,
                x1=mean_val,
                y1=curr_hist_ymax,
                line=dict(color="rgb(0, 0, 0)", width=6),
                row=idx + 1,
                col=1,
            )

            fig.add_annotation(
                x=mean_val,
                y=curr_hist_ymax,
                text=f"μ={mean_val:.2f}<br>σ={std_val:.2f}",
                font=dict(size=12, color="black"),
                bgcolor="rgb(255,255,255)",
                row=idx + 1,
                col=1,
            )

    # STANDARDIZER.standardize(fig)

    fig.update_layout(height=1800, title_text=title)
    fig.update_xaxes(title_text=xaxis_title, range=[0, 120], row=4, col=1)
    fig.update_yaxes(
        range=[0, hist_ymax],
        title_text=DENSITY_TYPE.capitalize() if DENSITY else "Count",
    )

    fig.write_image(os.path.join(output_dir, f"{file_name}.png"))

    BINS = 150

    # filter for generations > 0 and < 5
    filtered_data = data[data["generation"] < 5]

    # standardizer = PlotlyFormatter(
    #     title_font={"size": 48, "family": "Times New Roman", "color": "black"},
    #     axis_title_ratio=1.25,
    #     tick_font_ratio=1.5,
    #     legend_font_ratio=1.25,
    #     subtitle_ratio=1.25,
    # )

    for dosage in filtered_data["dosage"].unique():
        fig = go.Figure()
        curr_data = filtered_data[filtered_data["dosage"] == dosage]
        hist_ymax = 0
        gen_max = 0
        y_max = 0

        for idx, generation in enumerate(
            curr_data["generation"].unique(), start=curr_data["generation"].min()
        ):
            gen_data = curr_data[curr_data["generation"] == generation]
            gen_data = gen_data[prop].dropna()

            if len(gen_data) <= 1:
                break

            gen_max = max(gen_max, generation)

            if only_kde:
                kde = gaussian_kde(gen_data, bw_method=0.1)
                x_vals = np.linspace(gen_data.min(), gen_data.max(), 100)
                if DENSITY:
                    y_vals = kde(x_vals)
                else:
                    y_vals = kde(x_vals) * len(gen_data) * (x_vals[1] - x_vals[0])

                # coordinates for the surface mesh
                x_top = [generation] * len(x_vals)
                y_top = x_vals.tolist()
                z_top = y_vals.tolist()

                x_bottom = [generation] * len(x_vals)
                y_bottom = x_vals.tolist()
                z_bottom = [0] * len(x_vals)

                x_mesh = x_top + x_bottom
                y_mesh = y_top + y_bottom
                z_mesh = z_top + z_bottom

                y_max = max(y_max, x_vals.max())

                # create the mesh
                faces = []
                n = len(x_vals)
                for i in range(n - 1):
                    faces.append((i, i + 1, i + n))
                    faces.append((i + 1, i + n + 1, i + n))

                i, j, k = zip(*faces)

                fig.add_trace(
                    go.Mesh3d(
                        x=x_mesh,
                        y=y_mesh,
                        z=z_mesh,
                        i=i,
                        j=j,
                        k=k,
                        color=COLOR_PALETTE.get(dosage, "gray"),
                        opacity=0.5,
                        showscale=False,
                    )
                )

                hist_ymax = max(hist_ymax, max(z_mesh))

            else:
                a0 = np.histogram(gen_data, bins=BINS, density=DENSITY)[0].tolist()
                a0 = np.repeat(a0, 2).tolist()
                a0.insert(0, 0)
                a0.pop()
                a0[-1] = 0

                a1 = np.histogram(gen_data, bins=BINS - 1, density=DENSITY)[1].tolist()
                a1 = np.repeat(a1, 2)

                verts, tri = triangulate_histogram([idx] * len(a0), a1, a0)
                x, y, z = verts.T
                I, J, K = tri.T

                hist_ymax = max(hist_ymax, z.max())

                fig.add_traces(
                    go.Mesh3d(
                        x=x,
                        y=y,
                        z=z,
                        i=I,
                        j=J,
                        k=K,
                        color=COLOR_PALETTE.get(dosage, "gray"),
                        opacity=0.7,
                    )
                )
            # kde
            if len(gen_data) < 2:
                raise Exception(
                    f"Not enough data for generation {generation} in dosage {dosage}"
                )

            kde = gaussian_kde(gen_data, bw_method=0.1)
            x_vals = np.linspace(gen_data.min(), gen_data.max(), 100)
            if DENSITY:
                y_vals = kde(x_vals)
            else:
                y_vals = kde(x_vals) * len(gen_data) * (x_vals[1] - x_vals[0])

            fig.add_traces(
                go.Scatter3d(
                    x=[generation] * len(x_vals),
                    y=x_vals,
                    z=y_vals,
                    mode="lines",
                    line=dict(color="black", width=6),
                    showlegend=False,
                )
            )

            # mean and std
            if show_mean:
                mean_val = gen_data.mean()
                std_val = gen_data.std()

                fig.add_traces(
                    go.Scatter3d(
                        x=[generation, generation],
                        y=[mean_val, mean_val],
                        z=[0, max(y_vals)],
                        mode="lines",
                        line=dict(color="black", width=15),
                        showlegend=False,
                    )
                )

                fig.add_trace(
                    go.Scatter3d(
                        x=[generation + 0.4],
                        y=[mean_val],
                        z=[max(y_vals)],
                        mode="text",
                        text=[f"µ={mean_val:.2f}<br>σ={std_val:.2f}"],
                        textfont=dict(size=34, family="Times New Roman", color="black"),
                        textposition="top center",
                        showlegend=False,
                    )
                )

        # standardizer.standardize(fig)

        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title="Generation",
                    range=[
                        filtered_data["generation"].min(),
                        gen_max + 1,
                    ],
                    tickmode="linear",
                    tickfont=dict(size=20, family="Times New Roman", color="black"),
                    title_font=TITLE_FONT,
                ),
                yaxis=dict(
                    title=xaxis_title,
                    tickfont=dict(size=20, family="Times New Roman", color="black"),
                    title_font=TITLE_FONT,
                    range=[0, y_max],
                ),
                zaxis=dict(
                    title="Density" if DENSITY else "Count",
                    tickfont=dict(size=20, family="Times New Roman", color="black"),
                    title_font=TITLE_FONT,
                    range=[0, hist_ymax],
                ),
            ),
            title_text=dosage,
            title_x=0.5,
            template="plotly",
            width=3200,
            height=2400,
        )

        fig.write_image(os.path.join(output_dir, f"{file_name}_{dosage}.png"))
        # fig.write_html(os.path.join(output_dir, f"{file_name}_{dosage}.html"))


def create_division_profile_plot(data, output_dir, num_frames, subsampling=100):
    # colors as rgb
    generation_colors = {
        0: "rgb(0, 0, 255)",  # blue
        1: "rgb(0, 128, 0)",  # green
        2: "rgb(255, 255, 0)",  # yellow
        3: "rgb(0, 0, 139)",  # darkblue
        4: "rgb(255, 165, 0)",  # orange
        5: "rgb(255, 192, 203)",  # pink
        6: "rgb(255, 0, 0)",  # red
        7: "rgb(128, 0, 128)",  # purple
        8: "rgb(165, 42, 42)",  # brown
        9: "rgb(128, 128, 128)",  # gray
    }

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True)

    y_max_lengths = []

    for idx, dosage in enumerate(["Control", "High"]):
        filtered_data = data[data["dosage"] == dosage].copy()
        legend_entries = {}

        if subsampling:
            # seed
            np.random.seed(17)

            unique_paths = filtered_data["path_index_new"].unique()

            if len(unique_paths) > subsampling:
                path_to_gen = (
                    filtered_data.groupby("path_index_new")["generations"]
                    .first()
                    .to_dict()
                )
                gen_counts = filtered_data["generations"].value_counts(normalize=True)
                probs = [gen_counts[path_to_gen[p]] for p in unique_paths]
                probs = np.array(probs) / np.sum(probs)

                sampled_paths = np.random.choice(
                    unique_paths, size=subsampling, replace=False, p=probs
                )
                filtered_data = filtered_data[
                    filtered_data["path_index_new"].isin(sampled_paths)
                ]

                remap = {
                    old_index: new_index
                    for new_index, old_index in enumerate(sorted(sampled_paths))
                }
                filtered_data["path_index_new"] = filtered_data["path_index_new"].map(
                    remap
                )
        else:
            filtered_data["path_index_new"] = pd.Categorical(
                filtered_data["path_index_new"]
            ).codes

        y_max_lengths.append(len(filtered_data["path_index_new"].unique()))

        for _, row in filtered_data.iterrows():
            gen = row["generation"]
            if row["fate"] == "apoptosis-extension":
                color = "black"
            else:
                color = generation_colors.get(gen, "lightgray")
                legend_entries[gen] = generation_colors.get(gen, "lightgray")

            fig.add_trace(
                go.Bar(
                    x=[row["length"]],
                    y=[row["path_index_new"]],
                    orientation="h",
                    base=row["start_time"],
                    marker_color=color,
                    showlegend=False,
                    width=1,
                ),
                row=1,
                col=idx + 1,
            )

        if idx == 0:
            for gen, color in sorted(legend_entries.items(), key=lambda x: -x[0]):
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],  # Invisible data
                        mode="markers",
                        marker=dict(color=color, size=10),
                        name=f"Generation {gen}",
                        showlegend=True,
                    ),
                    row=1,
                    col=2,
                )

            if (filtered_data["fate"] == "apoptosis-extension").any():
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        marker=dict(color="black", size=10),
                        name="Apoptosis",
                        showlegend=True,
                    ),
                    row=1,
                    col=2,
                )

        fig.add_annotation(
            x=96,
            # y=-4,
            y=1.018,
            xref=f"x{idx + 1}",
            # yref="y",
            yref="paper",
            text="Treatment" if idx == 1 else "Untreated",
            showarrow=False,
            font=dict(size=34, family="Times New Roman", color="black", weight="bold"),
        )

        fig.add_trace(
            go.Scatter(
                x=[96, 96],
                y=[0, len(filtered_data["path_index_new"].unique())],
                mode="lines",
                line=dict(color="red", width=4, dash="dash"),
                name="Treatment Line",
                showlegend=False,
            ),
            row=1,
            col=idx + 1,
        )

        # STANDARDIZER.standardize(fig)

        fig.update_layout(barmode="stack")

    fig.update_xaxes(
        title="Time Frames",
        title_font=dict(size=34, family="Times New Roman", color="black"),
        tickfont_size=28,
        range=[0, num_frames],
    )

    for idy, y_max in enumerate(y_max_lengths):
        fig.update_yaxes(
            range=[y_max_lengths[idy], 0],
            title="Single Cells" if idy == 0 else None,
            title_font=dict(size=34, family="Times New Roman", color="black"),
            tickfont_size=28,
            row=1,
            col=idy + 1,
        )

    fig.add_annotation(
        x=-0.055,
        y=1.0275,
        text="(a)",
        showarrow=False,
        xref="paper",
        yref="paper",
        align="center",
        font=dict(size=34, family="Times New Roman", color="black", weight="bold"),
    )

    fig.add_annotation(
        x=0.52,
        y=1.0275,
        text="(b)",
        showarrow=False,
        xref="paper",
        yref="paper",
        align="center",
        font=dict(size=34, family="Times New Roman", color="black", weight="bold"),
    )

    fig.update_layout(
        margin=dict(r=10, t=50), width=1000, height=2000, legend_font_size=28
    )

    fig.layout.annotations[0].font = dict(
        family="Times New Roman", size=28, color="black", weight="bold"
    )
    fig.layout.annotations[1].font = dict(
        family="Times New Roman", size=28, color="black", weight="bold"
    )

    fig.update_layout(legend_itemsizing="constant")

    fig.write_image(os.path.join(output_dir, "division_profile_combined_new.eps"))


def create_division_profile_plot_old(data, output_dir, num_frames, subsampling=300):
    # colors as rgba
    generation_colors = {
        0: "rgba(0, 0, 255, 1)",  # blue
        1: "rgba(0, 128, 0, 1)",  # green
        2: "rgba(255, 255, 0, 1)",  # yellow
        3: "rgba(0, 0, 139, 1)",  # darkblue
        4: "rgba(255, 165, 0, 1)",  # orange
        5: "rgba(255, 192, 203, 1)",  # pink
        6: "rgba(255, 0, 0, 1)",  # red
        7: "rgba(128, 0, 128, 1)",  # purple
        8: "rgba(165, 42, 42, 1)",  # brown
        9: "rgba(128, 128, 128, 1)",  # gray
    }

    fig = make_subplots(rows=1, cols=2)

    for idx, dosage in enumerate(["Control", "High"]):
        filtered_data = data[data["dosage"] == dosage].copy()
        legend_entries = {}

        if subsampling:
            unique_paths = filtered_data["path_index_new"].unique()

            if len(unique_paths) > subsampling:
                sampled_paths = np.random.choice(unique_paths, size=subsampling)
                filtered_data = filtered_data[
                    filtered_data["path_index_new"].isin(sampled_paths)
                ]

                remap = {
                    old_index: new_index
                    for new_index, old_index in enumerate(sorted(sampled_paths))
                }
                filtered_data["path_index_new"] = filtered_data["path_index_new"].map(
                    remap
                )
        else:
            filtered_data["path_index_new"] = pd.Categorical(
                filtered_data["path_index_new"]
            ).codes

        for _, row in filtered_data.iterrows():
            gen = row["generation"]
            if row["fate"] == "apoptosis-extension":
                color = "black"
            else:
                color = generation_colors.get(gen, "lightgray")
                legend_entries[gen] = generation_colors.get(gen, "lightgray")

            fig.add_trace(
                go.Bar(
                    x=[row["length"]],
                    y=[row["path_index_new"]],
                    orientation="h",
                    base=row["start_time"],
                    marker_color=color,
                    showlegend=False,
                    width=1,
                ),
                row=1,
                col=idx + 1,
            )

        for gen, color in legend_entries.items():
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],  # Invisible data
                    mode="markers",
                    marker=dict(color=color, size=10),
                    name=f"Generation {gen}",
                    showlegend=True,
                ),
                row=1,
                col=idx + 1,
            )
        if (filtered_data["fate"] == "apoptosis-extension").any():
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(color="black", size=10),
                    name="Apoptosis-Extension",
                    showlegend=True,
                ),
                row=1,
                col=idx + 1,
            )

        fig.add_annotation(
            x=96,
            y=-5,
            xref="x",
            yref="paper",
            text="Treatment",
            showarrow=False,
            font=dict(size=120, family="Times New Roman", color="black"),
            row=1,
            col=idx + 1,
        )

        fig.add_shape(
            type="line",
            x0=96,
            x1=96,
            y0=0,
            y1=len(
                filtered_data["path_index_new"].unique()
            ),  # use paper coordinates for y
            xref="x",  # x stays in data coordinates
            yref="paper",  # y spans the whole plotting area
            line=dict(color="rgb(255, 0, 0)", width=6, dash="dash"),
            layer="above",  # draw above bars
            row=1,
            col=idx + 1,
        )

        # STANDARDIZER.standardize(fig)

        fig.update_xaxes(
            title="Time (0.5hr)",
            title_font=TITLE_FONT,
            tickfont=TICK_FONT,
            range=[0, num_frames],
            row=1,
            col=idx + 1,
        )
        fig.update_yaxes(
            title="Cell Division Path",
            autorange="reversed",
            title_font=TITLE_FONT,
            tickfont=TICK_FONT,
            row=1,
            col=idx + 1,
        )

        fig.update_layout(
            barmode="stack",
        )

    fig.update_layout(width=1400, height=3000)

    fig.write_image(os.path.join(output_dir, f"division_profile_test.png"))


def create_plotly_boxplot(
    data,
    title,
    y_data,
    xaxis_title,
    yaxis_title,
    boxpoints="outliers",
    boxmean=False,
    correlation=False,
    p_value=None,
    output_dir=None,
    file_name=None,
    save_html=False,
):
    import time

    import plotly.express as px

    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    fig.show()

    if not output_dir:
        tmp_dir = "test"
        os.makedirs(tmp_dir, exist_ok=True)
    else:
        tmp_dir = output_dir
    fig.write_image(os.path.join(tmp_dir, "random.eps"))

    # wait 5 seconds

    time.sleep(5)

    os.remove(os.path.join(tmp_dir, "random.eps"))

    legend_ranking = {"Control": 0, "Low": 1, "Medium": 2, "High": 3}

    fig = go.Figure()

    for dosage_level, idx in sorted(legend_ranking.items(), key=lambda item: item[1]):
        filtered_data = data[data["dosage"] == dosage_level]

        if len(filtered_data) == 0:
            continue

        fig.add_trace(
            go.Box(
                x=filtered_data["generation"],
                y=filtered_data[y_data],
                name=str(dosage_level),
                boxpoints=boxpoints,
                notched=False,
                marker_color=COLOR_PALETTE.get(dosage_level, "gray"),
                fillcolor=COLOR_PALETTE.get(dosage_level, "gray"),
                line_color="black",
                boxmean=boxmean,
                showlegend=True,
                legendrank=legend_ranking.get(dosage_level, 99),
            ),
        )

    fig.update_layout(boxmode="group")

    if correlation:
        for start, end, color, label in REGIONS:
            for sign in [-1, 1]:
                # for _row in [1, 3]:
                fig.add_shape(
                    type="rect",
                    x0=filtered_data["generation"].min() - 1.5,
                    x1=filtered_data["generation"].max(),
                    y0=start * sign,
                    y1=end * sign,
                    fillcolor=color,
                    opacity=0.35,
                    layer="below",
                    name=label,
                )

            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=10, color=color),
                    name=label,
                    legendgroup="group2",
                    legendgrouptitle_text="Interpretation",
                    showlegend=True,
                )
            )

        # for _row in [1, 3]:
        fig.update_yaxes(
            tickmode="linear",
            title_text=yaxis_title,
            tick0=-1,
            dtick=0.25,
        )

    else:
        fig.update_yaxes(
            title_text=yaxis_title,
        )

    fig.update_xaxes(
        title_text=xaxis_title,
        type="category",
    )

    # STANDARDIZER.standardize(fig)

    fig.update_layout(
        width=1200,
        height=600,
        legend=(
            dict(
                x=0,
                y=-0.2,
                orientation="h",
                indentation=15,
                itemsizing="constant",
                font=dict(size=34, family="Times New Roman", color="black"),
            )
        ),
        margin=dict(r=10, t=10),
    )

    fig.update_xaxes(
        title_font=dict(size=34, family="Times New Roman", color="black"),
        tickfont_size=28,
    )
    fig.update_yaxes(
        title_font=dict(size=34, family="Times New Roman", color="black"),
        tickfont_size=28,
    )

    if output_dir:
        fig.write_image(os.path.join(output_dir, f"{file_name}.eps"))
        if save_html:
            fig.write_html(os.path.join(output_dir, f"{file_name}.html"))
    else:
        return fig


def create_plotly_boxes(
    data,
    title,
    y_data,
    xaxis_title,
    yaxis_title,
    output_dir=None,
    file_name=None,
    save_html=False,
):
    import time

    import plotly.express as px

    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    fig.show()

    if not output_dir:
        tmp_dir = "test"
        os.makedirs(tmp_dir, exist_ok=True)
    else:
        tmp_dir = output_dir
    fig.write_image(os.path.join(tmp_dir, "random.eps"))

    # wait 5 seconds

    time.sleep(5)

    os.remove(os.path.join(tmp_dir, "random.eps"))

    box_width = 0.2
    num_groups = len(data["dosage"].unique())
    fig = go.Figure()

    for gen_idx, generation in enumerate(
        data["generation"].unique(), start=data["generation"].min()
    ):
        for dose_idx, dosage in enumerate(["Control", "Cisplatin"]):
            # for dose_idx, dosage in enumerate(data["dosage"].unique()):
            filtered_data = data[
                (data["generation"] == generation) & (data["dosage"] == dosage)
            ][y_data]

            if len(filtered_data) == 0:
                continue

            q1 = np.nanpercentile(filtered_data, 25)
            q3 = np.nanpercentile(filtered_data, 75)
            median = np.nanmedian(filtered_data)
            mean = np.nanmean(filtered_data)

            # x_center = _idx
            x_center = gen_idx + (dose_idx - (num_groups - 1) / 2) * box_width * 1.5

            # box
            fig.add_shape(
                type="rect",
                x0=x_center - box_width / 2,
                x1=x_center + box_width / 2,
                y0=q1,
                y1=q3,
                fillcolor=COLOR_PALETTE.get(dosage, "gray"),
                line=dict(color="black", width=1),
                legendgroup="group1",
                legendrank=1,
                name=str(dosage) if gen_idx == 1 else None,
                showlegend=gen_idx == 1,
                legend="legend",
                opacity=1,
                layer="between",
            )

            # mean circle
            fig.add_trace(
                go.Scatter(
                    x=[x_center],
                    y=[mean],
                    mode="markers",
                    marker=dict(color="red", size=8, symbol="diamond"),
                    legendgroup="group1",
                    name="Mean" if (gen_idx == 1 and dose_idx == 0) else None,
                    showlegend=(gen_idx == 1 and dose_idx == 0),
                    legend="legend",
                    hoverinfo="skip",
                    opacity=1,
                )
            )

            # median line
            fig.add_shape(
                type="line",
                x0=x_center - box_width / 2,
                x1=x_center + box_width / 2,
                y0=median,
                y1=median,
                legendgroup="group1",
                showlegend=(gen_idx == 1 and dose_idx == 0),
                legend="legend",
                name="Median" if (gen_idx == 1 and dose_idx == 0) else None,
                line=dict(color="black", width=4, dash="dot"),
                opacity=1,
            )

    for start, end, color, label in REGIONS:
        for sign in [-1, 1]:
            fig.add_shape(
                type="rect",
                x0=max(data["generation"].unique().min() - 2.5, -0.5),
                x1=5.5,
                y0=start * sign,
                y1=end * sign,
                fillcolor=color,
                opacity=0.35,
                layer="below",
                name=label,
            )

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=25, color=color),
                name=label,
                legendgroup="group2",
                showlegend=True,
                legend="legend2",
            )
        )

    fig.update_layout(
        xaxis=dict(
            title_text=xaxis_title,
            tickmode="array",
            tickvals=list(range(len(data["generation"].unique()))),
            ticktext=data["generation"].unique(),
        ),
        yaxis=dict(
            title_text=yaxis_title,
            tickmode="linear",
            tick0=-1,
            dtick=0.25,
        ),
    )

    # STANDARDIZER.standardize(fig)
    fig.update_xaxes(
        title_font=dict(size=34, family="Times New Roman", color="black"),
        tickfont_size=28,
    )
    fig.update_yaxes(
        title_font=dict(size=34, family="Times New Roman", color="black"),
        tickfont_size=28,
    )

    fig.update_layout(
        width=1200,
        height=600,
        legend=dict(
            orientation="h",
            y=-0.08,
            x=0,
            font=dict(size=34, family="Times New Roman", color="black"),
            traceorder="normal",
        ),
        legend2=dict(
            orientation="h",
            y=-0.2,
            x=0,
            font=dict(size=34, family="Times New Roman", color="black"),
            traceorder="normal",
        ),
        margin=dict(r=10, t=10),
    )

    if output_dir:
        fig.write_image(os.path.join(output_dir, f"{file_name}.eps"))
        if save_html:
            fig.write_html(os.path.join(output_dir, f"{file_name}.html"))
    else:
        return fig


def create_cell_cycle_corr_heatmap(results, name, output_dir=None):
    pivot = results.pivot(index="generation", columns="dosage", values="pearson_corr")

    # reset figure
    plt.clf()

    order = ["Control", "Low", "Medium", "High"]

    pivot = pivot[order]

    sns.heatmap(pivot, annot=True, cmap="coolwarm", center=0)

    if output_dir:
        plt.savefig(os.path.join(output_dir, f"cell_cycle_corr_{name}_heatmap.png"))
    else:
        return plt.gcf()


def create_plotly_lineplot(
    data,
    title,
    y_data,
    xaxis_title,
    yaxis_title,
    output_dir=None,
    file_name=None,
    save_html=False,
):
    fig = make_subplots(
        cols=4,
        rows=1,
        shared_yaxes=True,
        horizontal_spacing=0.01,
        subplot_titles=[k for k in COLOR_PALETTE.keys()],
    )

    colors = {
        "alive_fate_sync_proportion": "green",
        "div_fate_sync_proportion": "blue",
        "death_fate_sync_proportion": "red",
    }

    naming = {
        "alive_fate_sync_proportion": "Alive",
        "div_fate_sync_proportion": "Mitosis",
        "death_fate_sync_proportion": "Apoptosis",
    }

    for (dosage, fate_type), df in data.groupby(["dosage", "fate_type"]):
        fig.add_trace(
            go.Scatter(
                x=df["generation"],
                y=df[y_data],
                mode="lines+markers",
                name=f"{dosage} | {fate_type}",
                line=dict(color=colors.get(fate_type, "gray")),
                marker=dict(symbol="circle", size=8),
                showlegend=False,
            ),
            col=list(COLOR_PALETTE.keys()).index(dosage) + 1,
            row=1,
        )

    # Add dummy traces for legend
    for fate_type in colors.keys():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines+markers",
                name=naming.get(fate_type, fate_type),
                line=dict(color=colors.get(fate_type, "gray")),
                marker=dict(symbol="circle", size=8),
                showlegend=True,
            )
        )

    fig.update_xaxes(title_text="Generation", type="category")

    fig.update_traces(line=dict(width=3), marker=dict(size=8, symbol="circle"))

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis_type="category",
        legend_title_text="Fate",
    )

    # STANDARDIZER.standardize(fig)

    fig.update_layout(height=800)

    if output_dir:
        fig.write_image(os.path.join(output_dir, f"{file_name}.png"))
        if save_html:
            fig.write_html(os.path.join(output_dir, f"{file_name}.html"))
    else:
        return fig


def create_time_resolved_plot(data, output_dir):
    print("creating time-resolved plot...")
    print("Shape: ", data.shape)
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=data,
        x="time_index",
        y="area",
        hue="lineage_id",
        style="dosage",
        markers=True,
    )
    plt.title("Time-Resolved Cell Area Changes")
    plt.xlabel("Time (Frame Index)")
    plt.ylabel("Cell Area (Bounding Box Area)")
    plt.grid(True)
    output_file = os.path.join(output_dir, "time_resolved_plot.png")
    plt.savefig(output_file)
