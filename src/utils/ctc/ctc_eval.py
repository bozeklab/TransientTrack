"""
Evaluation script for the CTC Challenge. Software taken from http://celltrackingchallenge.net/evaluation-methodology/ and script partly adapted from https://github.com/kaloeffler/EmbedTrack/blob/master/embedtrack/ctc_metrics/eval_ctc.py
"""

import os
import platform
import re
import shutil
import subprocess
from os.path import isfile, join
from pathlib import Path

from utils import misc


def get_platform():
    """Select the platform the script is running on.

    Returns
    -------
    str
        The platform the script is running on. Possible values are "Win", "Linux" and "Mac".
    """

    sys_type = platform.system()
    if sys_type == "Windows":
        return "Win"
    elif sys_type == "Linux":
        return "Linux"
    elif sys_type == "Darwin":
        return "Mac"
    else:
        raise ValueError("Unknown system type: {}".format(sys_type))


def load_files(path: Path, regex, extension=""):
    """Loads all files in a given directory that match a given regex.

    Parameters
    ----------
    path : Path
        The path to the directory containing the files.
    regex : str
        The regex the files must match.
    extension : str, optional
        The extension the files must have. The default is "".

    Returns
    -------
    list
        List containing the names of the files that match the given regex.
    """
    files = [
        f
        for f in os.listdir(path)
        if isfile(join(path, f)) and re.match(regex, f) and f.endswith(extension)
    ]
    return files


def check_files(method, evaluation_dir, seq_id):
    """Checks whether the files needed for the evaluation are present and in the correct format.
    A description of the correct format can be found at http://celltrackingchallenge.net/evaluation-methodology/.

    Parameters
    ----------
    method : str
        Defines the method to be evaluated. Possible values are "DET", "SEG" and "TRA".
    evaluation_dir : Path
        Depicts the path to the directory containing the results and the ground truth.
    seq_id : str
        The sequence ID of the sequence to be evaluated.

    Returns
    -------
    bool
        True if all files are present, False otherwise.
    """

    folders_gt = [x for x in os.listdir(Path(evaluation_dir, seq_id + "_GT/"))]
    folders_res = [x for x in os.listdir(Path(evaluation_dir, seq_id + "_RES/"))]

    if method == "SEG":
        if "SEG" not in folders_gt and "SEG" not in folders_res:
            return None
        else:
            if not (
                Path(evaluation_dir, seq_id + "_GT/SEG/").exists()
                and Path(evaluation_dir, seq_id + "_RES/")
            ):
                return False

            files_gt = load_files(
                Path(evaluation_dir, seq_id + "_GT/SEG/"),
                r"^man_seg(?:0*(?:[0-9]{3}|[0-9]{4})\.tif|_0*(?:[0-9]{3})(?:_\d{3})?(\.tif)?)$",  # TODO: FIX
            )

            files_res = load_files(
                Path(evaluation_dir, seq_id + "_RES/"),
                r"^mask0*(?:[0-9]{3}|[0-9]{4})\.tif$",
            )

            return True if len(files_gt) > 0 and len(files_res) > 0 else False
    elif method == "DET":
        if os.path.exists(Path(evaluation_dir, seq_id + "_GT/TRA/")) and os.path.exists(
            Path(evaluation_dir, seq_id + "_RES/res_track.txt")
        ):
            files_gt = load_files(
                Path(evaluation_dir, seq_id + "_GT/TRA/"), r".*", extension=".txt"
            )

            files_res = os.path.isfile(
                Path(evaluation_dir, seq_id + "_RES/res_track.txt")
            )

            return True if len(files_gt) > 0 and files_res else False
        else:
            return False
    elif method == "TRA":
        if os.path.exists(Path(evaluation_dir, seq_id + "_GT/TRA/")) and os.path.exists(
            Path(evaluation_dir, seq_id + "_RES/")
        ):
            files_gt = load_files(
                Path(evaluation_dir, seq_id + "_GT/TRA/"),
                r"^man_track0*(?:[0-9]{3}|[0-9]{4})\.tif$",
            )

            files_res = load_files(
                Path(evaluation_dir, seq_id + "_RES/"),
                r"^mask*(?:[0-9]{3}|[0-9]{4})\.tif$",
            )

            track_file_gt = os.path.isfile(
                Path(evaluation_dir, seq_id + "_GT/TRA/man_track.txt")
            )

            track_file_res = os.path.isfile(
                Path(evaluation_dir, seq_id + "_RES/res_track.txt")
            )
            return (
                True
                if len(files_gt) > 0
                and len(files_res) > 0
                and track_file_res
                and track_file_gt
                else False
            )
        else:
            return False

    return False


def calc_score(
    method: str,
    evaluation_dir: Path,
    seq_id: int,
    platform: str,
    penalize_det=1,
):
    """Computes the score of a given method.

    Parameters
    ----------
    method : str
        Defines the method to be evaluated. Possible values are "DET", "SEG" and "TRA".
    evaluation_dir : Path
        Depicts the path to the directory containing the results and the ground truth.
    seq_id : int
        The sequence ID of the sequence to be evaluated.
    platform : str
        The platform the script is running on. Possible values are "Win", "Linux" and "Mac".
    penalize_det : int, optional
        Defines how extra detected cells are penalized. The default is 1.

    Returns
    -------
    float
        The score of the given method based on the results and the ground truth.
    """

    assert method.upper() in ["DET", "SEG", "TRA"], "Unknown method: {}".format(method)

    print("{} score calculation started...".format(method))

    exec_folder = Path(Path(__file__).parent, "CTC_eval", platform)

    assert Path.exists(exec_folder), "Evaluation executables not found."

    # get correct executable based on method
    measure_files = [
        file
        for file in os.listdir(exec_folder)
        if file.startswith(method.upper() + "Measure")
    ]

    assert (
        len(measure_files) == 1
    ), "Measure file not found or multiple executables of the specified method found."

    measure_file_path = Path(exec_folder, measure_files[0])

    if not check_files(method, evaluation_dir, seq_id):
        print(
            "Could not find required folders/files for evaluation of {}.".format(method)
        )
        return None

    ##########
    # TODO: hardcoded value specific for the used dataset
    num_digits = 3
    ##########

    if method.upper() == "DET":  # DET takes an additional, optional parameter
        metric_results = subprocess.Popen(
            [
                measure_file_path.as_posix(),
                evaluation_dir.as_posix(),
                seq_id,
                str(num_digits),
                str(penalize_det),
            ],
            stdout=subprocess.PIPE,
        )
    else:
        metric_results = subprocess.Popen(
            [
                measure_file_path.as_posix(),
                evaluation_dir.as_posix(),
                seq_id,
                str(num_digits),
            ],
            stdout=subprocess.PIPE,
        )

    tmp_result, tmp = metric_results.communicate()

    measure_result = re.findall(r"\d\.\d*", tmp_result.decode("utf-8"))

    assert len(measure_result) == 1, "Error during evaluation."

    measure_result = float(measure_result[0])

    print("Method: {}. Score: {}".format(method, measure_result))

    return measure_result


def ctc_score(results_dir, gt_dir, evaluation_dir, subfolder):
    """Computes all CTC measures (DET, SEG, TRA) using the evaluation script provided by the CTC.

    Parameters
    ----------
    results_dir : str
        Path to the results' directory. The directory must contain the results in the format specified by the CTC.
    gt_dir : str
        Path to the ground truth directory. The directory must contain the ground truth in the format specified by the CTC.
    subfolder : str
        The subfolder in the results and ground truth directories.
    delete_tmp_files : bool, optional
        Defines whether temporary files should be deleted after the evaluation. The default is False.
    """

    platform = get_platform()

    # move results and ground truth to the same folder, since the evaluation software expects them to be in the same folder

    data_dir = gt_dir.parent

    if (
        results_dir.parent.as_posix() != data_dir.as_posix()
    ):  # results are not in the same folder as the ground truth. Copying both to a temporary folder
        gt_eval_dir = Path(evaluation_dir, subfolder + "_GT/TRA/")
        results_eval_dir = Path(evaluation_dir, subfolder + "_RES/")
        # remove existing files from tmp folder
        if os.path.exists(evaluation_dir):
            shutil.rmtree(evaluation_dir, onerror=misc.remove_readonly)

        # move results to tmp evaluation folder
        shutil.copytree(
            results_dir.as_posix(), results_eval_dir.as_posix(), dirs_exist_ok=True
        )
        shutil.copytree(gt_dir.as_posix(), gt_eval_dir.as_posix(), dirs_exist_ok=True)
    else:
        evaluation_dir = data_dir

    ctc_score = dict()
    ctc_score["DET"] = calc_score("DET", evaluation_dir, subfolder, platform)
    ctc_score["TRA"] = calc_score("TRA", evaluation_dir, subfolder, platform)

    return ctc_score
