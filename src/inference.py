import argparse
import os
import time

import cv2
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file
from tqdm import tqdm

from dataset.dataset import InferenceDataset
from models.defdetr import DeformableDetr
from utils.matching import (
    BYTETrackMatcher,
    DeepSORTTrackMatcher,
    SORTTrackMatcher,
    matching_TT,
)
from utils.misc import write_tracks
from utils.tracking_utils import (
    Tracklet,
    annotate_image,
    export_summary,
    get_centroids,
)
from utils.transforms import Normalize, StatefulChannelOscillation


def inference(
    detector_path,
    dataloader,
    patience_range,
    config,
    label_children=True,
    save_images=False,
    inf_dir=None,
    metric=None,
    eval_dir=None,
    start_id=None,
    matching_alg="tt",
):
    ############
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    iou_threshold = config.misc.iou_threshold
    low_conf_upper = config.misc.low_conf_upper
    low_conf_lower = config.misc.low_conf_lower
    img_type = config.misc.img_type

    runs = 1
    inference_times = []

    for _ in range(runs):

        if matching_alg == "tt":
            matching = matching_TT
        elif matching_alg == "SORT":
            matching = SORTTrackMatcher()
        elif matching_alg == "DeepSORT":
            matching = DeepSORTTrackMatcher()
        elif matching_alg == "ByteTrack":
            matching = BYTETrackMatcher()
        else:
            raise ValueError(f"Unsupported matching algorithm: {matching_alg}")

        num_channels = 3 if img_type == "rgb" else 1

        detector = DeformableDetr(
            train=False,
            batch_size=1,
            num_channels=num_channels,
            two_stage=config.model.two_stage,
            box_refine=config.model.box_refine,
        )
        state_dict = load_file(detector_path, device=str(device))
        detector.load_state_dict(state_dict, strict=True)
        detector.to(device)
        detector.eval()

        if metric is not None:
            metric = metric.to(device)

        tracklets = []
        patience_dict = dict()
        mitosis = dict()

        start_time = time.perf_counter()

        for time_step, (img, annotations) in tqdm(
            enumerate(dataloader, start=start_id)
        ):
            hs_curr_centroids, ls_curr_centroids = get_centroids(
                img,
                time_step,
                detector,
                device,
                low_conf_upper,
                low_conf_lower,
                iou_threshold,
                metric,
                annotations,
            )

            if matching_alg == "tt":
                if time_step == start_id:
                    for _id, cell in hs_curr_centroids.items():
                        tracklet = Tracklet(_id, time_step)
                        tracklet.add_cell(time_step, cell)
                        tracklets.append(tracklet)

                    last_new_label_index = (
                        max(hs_curr_centroids.keys()) if hs_curr_centroids.keys() else 0
                    )
                else:
                    (
                        tracklets,
                        last_new_label_index,
                        hs_curr_centroids,
                        patience_dict,
                        mitosis,
                    ) = matching(
                        time_step,
                        hs_curr_centroids,
                        ls_curr_centroids,
                        prev_centroids,
                        last_new_label_index,
                        tracklets,
                        patience_dict,
                        mitosis,
                        patience_range,
                    )

                prev_centroids = hs_curr_centroids
            else:
                matching(time_step, hs_curr_centroids, ls_curr_centroids)

        end_time = time.perf_counter()

        inference_times.append(end_time - start_time)

    # inference_time = end_time - start_time
    avg_inference_time = sum(inference_times) / len(inference_times)
    print(f"Mean execution times for {runs} runs: ", avg_inference_time)

    if matching_alg != "tt":
        # post-processing needed
        tracklets = matching.results

    if save_images:
        print("\nSaving images...")
        # reset stateful transforms to ensure correct visualization of the entire sequence
        corruption = False
        for transform in dataloader.dataset.transform.transforms:
            if isinstance(transform, StatefulChannelOscillation):
                transform._reset()
                corruption = True

            if isinstance(transform, Normalize):
                mean = transform.mean
                std = transform.std

        for time_step, (image, _) in tqdm(enumerate(dataloader, start=start_id)):
            candidates = [
                tracklet.get_cell(time_step)
                for tracklet in tracklets
                if tracklet.get_cell(time_step) is not None
            ]

            image = image.squeeze().numpy()

            img = annotate_image(
                image,
                img_type,
                candidates,
                mitosis,
                label_children,
                corruption=corruption,
                mean=mean,
                std=std,
            )

            if img_type == "rgb":
                img = img[:, :, ::-1]

            cv2.imwrite(os.path.join(inf_dir, "img_{:03d}.tif".format(time_step)), img)

        #  create video from images
        print("\nCreating video from images...")
        images = sorted(img for img in os.listdir(inf_dir) if img.endswith(".tif"))

        first_frame = cv2.imread(os.path.join(inf_dir, images[0]), -1)
        height, width, layers = first_frame.shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(
            os.path.join(inf_dir, "video.mp4"), fourcc, 6, (width, height)
        )

        for image in images:
            img_path = os.path.join(inf_dir, image)
            frame = cv2.imread(img_path, -1)
            video.write(frame)

        video.release()

    if eval_dir:
        export_summary(
            tracklets,
            mitosis,
            eval_dir,
        )

        write_tracks(
            tracklets,
            eval_dir,
            start_id=start_id,
        )

    return tracklets, mitosis, avg_inference_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default="model/model.safetensors",
        type=str,
    )

    parser.add_argument("--config", default="model/config.yaml", type=str)
    parser.add_argument(
        "--videos",
        default="src/data/dataset/",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="src/data/analysis",
    )
    parser.add_argument("--save-images", action="store_true", default=False)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    if args.save_images:
        inf_base_dir = os.path.join(args.output_dir, "inference")
        os.makedirs(inf_base_dir, exist_ok=True)
    else:
        inf_base_dir = None

    eval_base_dir = os.path.join(args.output_dir, "evaluation")
    os.makedirs(eval_base_dir, exist_ok=True)

    videos = sorted(
        [
            os.path.join(args.videos, f)
            for f in os.listdir(args.videos)
            if os.path.isdir(os.path.join(args.videos, f))
        ]
    )

    for video in tqdm(videos):
        video_dataset = InferenceDataset(
            data_dir=video,
            mean=config.training.mean,
            std=config.training.std,
        )

        eval_dir = os.path.join(eval_base_dir, video.split("/")[-1])
        os.makedirs(eval_dir, exist_ok=True)

        if args.save_images:
            inf_dir = os.path.join(inf_base_dir, video.split("/")[-1])
            os.makedirs(inf_dir, exist_ok=True)
        else:
            inf_dir = None

        inference(
            detector_path=args.checkpoint,
            inf_dir=inf_dir,
            eval_dir=eval_dir,
            dataloader=video_dataset,
            patience_range=5,
            config=config,
            save_images=args.save_images,
            label_children=False,
            start_id=1,
        )
