import argparse
import os

import pandas as pd
from omegaconf import OmegaConf
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from dataset.dataloader import CocoVidDataLoader
from dataset.dataset import CocoVideoDataset
from inference import inference
from utils.evaluation_utils import (
    apoptosis_evaluation,
    combine_mask_labels,
    ctc_evaluation,
    ctc_gt_origin,
    log_motmetrics,
    trackeval_evaluation,
    write_ctc_files,
)


def eval_videos(
    config,
    checkpoint,
    output_dir,
    dataloader,
    matching_alg,
    experiment_name=None,
    logger=None,
    videos=None,
    save_images=False,
    patience_range=5,
    delta=5,
    window=2,
):
    calc_ctc = True if matching_alg == "tt" else False

    mota_metric_keys = ["HOTA", "DetA", "AssA", "MOTA", "MOTP", "IDF1"]
    ctc_metric_keys = ["DET", "TRA", "LNK"]

    columns = (
        ["video_id", "num_tracks"]
        + mota_metric_keys
        + (ctc_metric_keys if calc_ctc else [])
        + ["mAP", "AP50", "AP_alive", "AP_dead"]
        + [
            "Precision",
            "Recall",
            "F1",
            "TP",
            "FP",
            "FN",
            "TN",
            "Mean-Error",
            "Timing Errors",
            "Timing Excluded",
        ]
        + (
            [
                "DIV_nGT",
                "DIV_nPred",
                "DIV_WC",
                "DIV_TP_i0",
                "DIV_TP_i1",
                "DIV_TP_i2",
            ]
            if calc_ctc
            else []
        )
        + ["Inference-Time"]
    )
    results = []

    wb_mapResults = []
    wb_ctcResults = []
    wb_motResults = []

    videos = videos or dataloader.tracking_videos

    for video in tqdm(videos):
        print(f"Processing video {video}")

        video_dataset = CocoVideoDataset(
            dataloader.coco,
            dataloader.data_dir,
            dataloader.start_id,
            dataloader.end_id,
            [video],
            dataloader.transform,
            channel=dataloader.channel,
        )

        video_dataloader = dataloader.tracking_dataloader(video_dataset)

        mot_sums, ctc_results, map_results, apoptosis_results, inf_time = evaluate(
            config=config,
            data_dir=dataloader.data_dir,
            output_dir=output_dir,
            position=video,
            save_images=save_images,
            label_children=False,
            label_segmentations=True,
            detector_path=checkpoint,
            dataloader=video_dataloader,
            patience_range=patience_range,
            start_id=dataloader.start_id,
            matching_alg=matching_alg,
            coco=dataloader.coco,
            delta=delta,
            window=window,
        )
        if mot_sums is not None:
            wb_motResults.append([mot_sums])

            if calc_ctc:
                wb_ctcResults.append(
                    [ctc_results["DET"], ctc_results["TRA"], ctc_results["LNK"]]
                )

            results.append(
                [video, mot_sums["NUM_TRACKS"]]
                + [mot_sums[k] for k in mota_metric_keys]
                + (
                    [
                        ctc_results["DET"].item(),
                        ctc_results["TRA"].item(),
                        ctc_results["LNK"].item(),
                    ]
                    if calc_ctc
                    else []
                )
                + [
                    map_results["map"].item(),
                    map_results["map_50"].item(),
                    map_results["map_per_class"][0].item(),
                    map_results["map_per_class"][1].item(),
                ]
                + [
                    apoptosis_results["event_precision"],
                    apoptosis_results["event_recall"],
                    apoptosis_results["event_f1"],
                    apoptosis_results["TP"],
                    apoptosis_results["FP"],
                    apoptosis_results["FN"],
                    apoptosis_results["TN"],
                    apoptosis_results["mean_timing_error"],
                    apoptosis_results["timing_errors"],
                    apoptosis_results["n_timing_excluded"],
                ]
                + (
                    [
                        ctc_results["DIV_nGT"].item(),
                        ctc_results["DIV_nPred"].item(),
                        ctc_results["DIV_WC"].item(),
                        ctc_results["DIV_TP_i0"].item(),
                        ctc_results["DIV_TP_i1"].item(),
                        ctc_results["DIV_TP_i2"].item(),
                    ]
                    if calc_ctc
                    else []
                )
                + [inf_time]
            )

        if map_results is not None:
            wb_mapResults.append([map_results["map"].item()])

    if logger:
        wb_mapResults = pd.DataFrame(wb_mapResults, columns=["map"])
        map_average = wb_mapResults.mean().to_frame().T

        if calc_ctc:
            wb_ctcResults = pd.DataFrame(wb_ctcResults, columns=["DET", "TRA"])
            ctc_average = wb_ctcResults.mean().to_frame().T

            logger.log_metrics({"Tracking/CTC": wb_ctcResults})
            logger.log_metrics({"Tracking/CTC_Average": ctc_average})

        logger.log_metrics({"Tracking/mAP": wb_mapResults})
        logger.log_metrics({"Tracking/mAP_Average": map_average})

        log_motmetrics(logger, wb_motResults)

        logger.finalize("success")

    experiment_name = experiment_name or "overall.csv"

    results = pd.DataFrame(results, columns=columns)
    results.to_csv(os.path.join(output_dir, experiment_name), index=False)


def evaluate(
    config,
    data_dir,
    output_dir,
    position,
    patience_range,
    matching_alg="tt",
    start_id=None,
    save_images=False,
    label_children=True,
    label_segmentations=False,
    detector_path=None,
    dataloader=None,
    coco=None,
    delta=5,
    window=2,
):
    calc_ctc = True if matching_alg == "tt" else False

    max_dets = 300

    if max_dets > 100:
        backend = "faster_coco_eval"
    else:
        backend = "pycocotools"

    mAP = MeanAveragePrecision(
        box_format="cxcywh",
        iou_type="bbox",
        class_metrics=True,
        extended_summary=False,
        backend=backend,
        max_detection_thresholds=[1, 10, max_dets],
    )

    eval_dir = os.path.join(output_dir, "evaluation", position)
    ctc_files = os.path.join(eval_dir, "ctc_files")
    os.makedirs(ctc_files, exist_ok=True)

    if len(os.listdir(ctc_gt)) <= 1:
        print("Skipping position", position, "as no ground truth is available.")
        return None, None, None, None, None

    ctc_gt = os.path.join(data_dir, position, "ctc_files")
    ctc_origin = ctc_gt_origin(ctc_gt)

    if save_images:
        inf_dir = os.path.join(output_dir, "inference", position)
        os.makedirs(inf_dir, exist_ok=True)
    else:
        inf_dir = None

    tracklets, mitosis, inf_time = inference(
        detector_path=detector_path,
        inf_dir=inf_dir,
        eval_dir=eval_dir,
        config=config,
        dataloader=dataloader,
        label_children=label_children,
        save_images=save_images,
        metric=mAP,
        patience_range=patience_range,
        start_id=start_id,
        matching_alg=matching_alg,
    )

    if tracklets:
        mot_sums, gt = trackeval_evaluation(
            tracklets,
            dataloader.dataset.start_id,
            dataloader.dataset.end_id,
            data_dir,
            eval_dir,
            position,
            config.training.resize_ratio,
        )

        # apoptosis evaluation

        # get video annotations
        video_id = [
            video_id
            for video_id, video_inf in coco.videos.items()
            if video_inf["name"] == position
        ]
        if len(video_id) > 0:
            video_id = video_id[0]

        apoptosis_sum = apoptosis_evaluation(
            coco,
            tracklets,
            video_id,
            dataloader.dataset.start_id,
            dataloader.dataset.end_id,
            delta,
            window=window,
        )
    else:
        mot_sums = None

    # return mot_sums, None, None

    if calc_ctc:
        write_ctc_files(
            tracklets,
            eval_dir,
            start_id=dataloader.dataset.start_id,
            end_id=dataloader.dataset.end_id,
            ctc_origin=ctc_origin,
        )

    if label_segmentations and calc_ctc:
        combine_mask_labels(
            tracklets,
            start_time=dataloader.dataset.start_id,
            end_time=dataloader.dataset.end_id,
            ctc_origin=ctc_origin,
            resize_ratio=config.training.resize_ratio,
            output_dir=ctc_files,
        )

        ctc_results = ctc_evaluation(ctc_files, ctc_gt, position, False)
        # combine both dicts
        # combined = {**ctc_results, **chota_score}
        ctc_results = pd.DataFrame.from_dict(ctc_results, orient="index").T
        ctc_results.to_csv(os.path.join(eval_dir, "eval_ctc.csv"), sep=";", index=None)
    else:
        ctc_results = None

    mAP = mAP.to("cuda:0")
    mAP = mAP.compute()

    print("mAP:", mAP["map"].item(), "\n")
    # print("mAP per class: ", mAP["map_per_class"], "\n")
    # print(mAP)

    return mot_sums, ctc_results, mAP, apoptosis_sum, inf_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", default="annotations.json", type=str)
    parser.add_argument(
        "--checkpoint",
        default="model/model.safetensors",
        type=str,
    )
    parser.add_argument("--config", default="model/config.yaml", type=str)
    parser.add_argument("--data_dir", default="src/data/dataset", type=str)
    parser.add_argument("--start_id", default=49, type=int)
    parser.add_argument("--end_id", default=234, type=int)
    parser.add_argument("--output_dir", default="src/data/output", type=str)
    parser.add_argument("--ban_list", type=str)
    parser.add_argument(
        "--matching_alg",
        default="tt",
        choices=["tt", "SORT", "DeepSORT", "ByteTrack"],
        type=str,
    )
    parser.add_argument("--experiment_name", default="overall", type=str)

    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    if "rgb" in config.misc.img_type:
        config.misc.channel = None

    dataloader = CocoVidDataLoader(
        data_dir=args.data_dir,
        ann_file=args.coco,
        img_type=config.misc.img_type,
        start_id=args.start_id,
        end_id=args.end_id,
        channel=config.misc.channel,
        resize_ratio=config.training.resize_ratio,
        amp=0,
    )

    videos = [f for f in dataloader.coco.dataset["info"]["tracking_videos"]]

    if args.ban_list:
        ban_list = pd.read_csv(
            args.ban_list,
            header=None,
            dtype=str,
        )
        ban_list = ban_list[0].values.tolist()

        videos = [video for video in videos if video not in ban_list]

    experiment_name = args.experiment_name

    eval_videos(
        config,
        args.checkpoint,
        args.output_dir,
        dataloader,
        experiment_name=f"{experiment_name}.csv",
        videos=videos,
        save_images=False,
        patience_range=5,
        matching_alg=args.matching_alg,
        delta=5,
        window=2,
    )
