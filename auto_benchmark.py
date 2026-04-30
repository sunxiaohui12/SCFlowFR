import json, os
from argparse import ArgumentParser
from os import path as osp
import numpy as np
import pandas as pd
import pyiqa
import torch
from pytorch_fid import fid_score
from flowsr.metrics import batched_iqa
from flowsr.io_utils import read_images_parallel as read_images
from flowsr.data.utils import list_image_files as load_file_list
from natsort import natsorted
from datetime import datetime
import gc


def get_method_list(root, dataset):
    method_dir = osp.join(root, dataset)
    if not osp.exists(method_dir):
        return []
    # List all entries in the dataset directory and filter for directories
    method_list = [
        name for name in os.listdir(method_dir) if osp.isdir(osp.join(method_dir, name))
    ]
    return natsorted(method_list)


# Device configuration
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def calc_fid(path_0, path_1):
    print(path_0, path_1)
    return fid_score.calculate_fid_given_paths(
        [path_0, path_1], batch_size=16, device=device, dims=2048, num_workers=8
    )


# Metrics configuration
METRICS_CONFIG = {
    "pytorch_fid": {
        "metric": calc_fid,
        "requires_ref": True,
        "path_based": True,
    },
    "fid": {
        "metric": pyiqa.create_metric("fid"),
        "requires_ref": True,
        "path_based": True,
    },
    # "psnr": {
    #     "metric": pyiqa.create_metric("psnr"),
    #     "requires_ref": True,
    #     "path_based": False,
    # },
    # "ssim": {
    #     "metric": pyiqa.create_metric("ssim"),
    #     "requires_ref": True,
    #     "path_based": False,
    # },
    # "lpips": {
    #     "metric": pyiqa.create_metric("lpips"),
    #     "requires_ref": True,
    #     "path_based": False,
    # },
    "niqe": {
        "metric": pyiqa.create_metric("niqe"),
        "requires_ref": False,
        "path_based": False,
    },
    # "ManIQA": {
    #     "metric": pyiqa.create_metric("maniqa"),
    #     "requires_ref": False,
    #     "path_based": False,
    # },
    # "CLIPIQA": {
    #     "metric": pyiqa.create_metric("clipiqa"),
    #     "requires_ref": False,
    #     "path_based": False,
    # },
    "musiq": {
        "metric": pyiqa.create_metric("musiq"),
        "requires_ref": False,
        "path_based": False,
    },
    "brisque": {
        "metric": pyiqa.create_metric("brisque"),
        "requires_ref": False,
        "path_based": False,
    },
}


def uint2single(img: np.ndarray) -> np.ndarray:
    """Convert uint8 to float32"""
    assert img.dtype == np.uint8
    return np.float32(img / 255.0)


def writeRslt(rsltPath, rslt):
    with open(osp.join(rsltPath, "rslt.txt"), "w") as f:
        f.write(json.dumps(rslt))


def readRslt(rsltPath):
    with open(osp.join(rsltPath, "rslt.txt"), "r") as f:
        rslt = json.load(f)
    return rslt


def optimized_tensor_conversion(image_list, device):
    # Preallocate numpy array with correct dimensions
    num_images = len(image_list)
    if num_images == 0:
        return torch.empty(0, device=device)

    h, w, c = image_list[0].shape
    np_arr = np.empty((num_images, c, h, w), dtype=np.float32)

    # Parallel conversion using numba or manual threading
    for i in range(num_images):
        np_arr[i] = (image_list[i].transpose(2, 0, 1) / 255.0).astype(np.float32)

    # Create pinned tensor and transfer asynchronously
    tensor = torch.from_numpy(np_arr).pin_memory()
    return tensor.to(device, non_blocking=True)


def calculate_metrics(methodName, degradName, rsltPath, hrPath, hrs, rslts):
    """Calculate all configured metrics"""
    rslt = {
        "methodName": methodName,
        "degradName": degradName,
        "calculated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Convert images to tensors
    # hr_tensor = torch.from_numpy(uint2single(np.array(hrs).transpose(0, 3, 1, 2))).to(
    #     device
    # )
    # rslt_tensor = torch.from_numpy(
    #     uint2single(np.array(rslts).transpose(0, 3, 1, 2))
    # ).to(device)
    hr_tensor = optimized_tensor_conversion(hrs, device)
    rslt_tensor = optimized_tensor_conversion(rslts, device)

    # Calculate metrics
    for metric_name, config in METRICS_CONFIG.items():
        print(f"calculating {metric_name.upper()} ...")
        if metric_name == "pytorch_fid":
            metric = config["metric"]
            hrPath = "load/CelebA-Test/HQ"
            rslt[metric_name] = metric(rsltPath, hrPath)
        elif metric_name == "fid":
            metric = config["metric"]
            rslt[metric_name] = metric(rsltPath, dataset_name="FFHQ", dataset_res=512, dataset_split="trainval70k")
        elif config["path_based"]:
            metric = config["metric"].to(device)
            hrPath = "load/CelebA-Test/HQ"
            rslt[metric_name] = metric(rsltPath, hrPath)
        else:
            metric = config["metric"].to(device)
            rslt[metric_name] = (
                batched_iqa(
                    metric,
                    rslt_tensor,
                    hr_tensor if config["requires_ref"] else None,
                    desc=f"calculating {metric_name.upper()}: ",
                )
                .mean()
                .item()
            )

    # Clear GPU memory
    if torch.cuda.is_available():
        hr_tensor = None
        rslt_tensor = None
        torch.cuda.empty_cache()
        gc.collect()

    return rslt


def handleDataset(dataset, methodList):
    rsltList = []
    hrPath = osp.join("load/benchmark", dataset, "HQ")
    print("loading HQ images...")
    hrs = read_images(natsorted(load_file_list(hrPath)))

    for methodName in methodList:
        if not osp.exists(osp.join(root, dataset, methodName)):
            continue
        degradationList = natsorted(os.listdir(osp.join(root, dataset, methodName)))
        for degradName in degradationList:
            print("==" * 20)
            print(f"{dataset} | {methodName} | {degradName}")
            rsltPath = osp.join(root, dataset, methodName, degradName)
            if not osp.exists(rsltPath):
                continue
            elif osp.exists(osp.join(rsltPath, "rslt.txt")) and (not force_recalc):
                rslt = readRslt(rsltPath)
                rslt["methodName"] = methodName
                rslt["degradName"] = degradName
            else:
                print(f"loading SR results from {rsltPath}...")
                rsltPathList = natsorted(load_file_list(rsltPath))
                rslts = read_images(rsltPathList)
                rslt = calculate_metrics(
                    methodName, degradName, rsltPath, hrPath, hrs, rslts
                )

            rslt["dataset"] = dataset
            rsltList.append(rslt)
            writeRslt(rsltPath, rslt)
            print(rslt)

    return rsltList


if __name__ == "__main__":
    root = "logs/_results"
    datasets = ["CelebChild","lfw","WebPhoto"]
    parser = ArgumentParser()
    parser.add_argument(
        "--force_recalc",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()
    force_recalc = args.force_recalc
    rslt_list = []
    for dataset in datasets:
        methodList = get_method_list(root, dataset)
        print(methodList)
        rslt_list.extend(handleDataset(dataset, methodList))
    df = pd.DataFrame(rslt_list)
    df.to_excel(
        osp.join(root, "result_all_real.xlsx"),
        index=False,
    )