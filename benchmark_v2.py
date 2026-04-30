import os
import json
from argparse import ArgumentParser
from os import path as osp
from datetime import datetime
from natsort import natsorted

import numpy as np
import torch
import pyiqa
from pytorch_fid import fid_score
import pandas as pd
import cv2

from flowsr.metrics import batched_iqa
from flowsr.dataset.utils import list_image_files as load_file_list
from flowsr.utils.io_utils import read_images_parallel as read_images


def get_device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def calc_fid(dir_or_stats_a: str, dir_or_stats_b: str, batch_size: int = 16, dims: int = 2048, num_workers: int = 8) -> float:
    return fid_score.calculate_fid_given_paths([dir_or_stats_a, dir_or_stats_b], batch_size=batch_size, device=get_device(), dims=dims, num_workers=num_workers)


def safe_image_processing(image, target_size=None):
    """安全地处理图像，处理各种异常情况"""
    if image is None:
        return None
    
    # 检查图像是否为空
    if image.size == 0:
        return None
    
    # 检查图像维度
    if len(image.shape) != 3 or image.shape[2] not in [1, 3, 4]:
        print(f"Warning: Invalid image shape {image.shape}")
        return None
    
    # 如果是灰度图像，转换为RGB
    if image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # 调整大小
    if target_size is not None:
        target_h, target_w = target_size
        try:
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        except Exception as e:
            print(f"Error resizing image: {e}")
            return None
    
    return image


def optimized_tensor_conversion(image_list, device: torch.device, target_size=None) -> torch.Tensor:
    num_images = len(image_list)
    if num_images == 0:
        return torch.empty(0, device=device)

    # 安全处理所有图像
    valid_images = []
    for i, img in enumerate(image_list):
        processed_img = safe_image_processing(img, target_size)
        if processed_img is not None:
            valid_images.append(processed_img)
        else:
            print(f"Warning: Skipping invalid image at index {i}")
    
    if len(valid_images) == 0:
        print("Warning: All images are invalid, returning empty tensor")
        return torch.empty(0, device=device)
    
    num_images = len(valid_images)
    h, w, c = valid_images[0].shape

    np_arr = np.empty((num_images, c, h, w), dtype=np.float32)
    for i in range(num_images):
        try:
            np_arr[i] = (valid_images[i].transpose(2, 0, 1) / 255.0).astype(np.float32)
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            # 创建一个默认的黑色图像
            np_arr[i] = np.zeros((c, h, w), dtype=np.float32)

    tensor = torch.from_numpy(np_arr).pin_memory()
    return tensor.to(device, non_blocking=True)


def compute_no_ref_metrics(sr_dir: str, fid_ref_dir: str = None, batch_size: int = 50, fid_ref_stats: str = None) -> dict:
    device = get_device()

    # Collect SR files
    sr_files = natsorted(load_file_list(sr_dir))
    if len(sr_files) == 0:
        raise FileNotFoundError(f"No images found in SR directory: {sr_dir}")

    result = {
        "sr_dir": sr_dir,
        "fid_ref_dir": fid_ref_dir if fid_ref_dir else None,
        "fid_ref_stats": fid_ref_stats if fid_ref_stats else None,
        "calculated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # FID计算
    if fid_ref_stats is not None:
        if not osp.isfile(fid_ref_stats):
            raise FileNotFoundError(f"FID reference stats (.npz) not found: {fid_ref_stats}")
        print("calculating FID (SR vs reference stats) ...")
        result["fid"] = float(calc_fid(sr_dir, fid_ref_stats))
        print(f"FID: {result['fid']:.4f}")
    elif fid_ref_dir is not None:
        if not osp.isdir(fid_ref_dir):
            raise FileNotFoundError(f"FID reference directory not found: {fid_ref_dir}")
        print("calculating FID (SR vs reference directory) ...")
        result["fid"] = float(calc_fid(sr_dir, fid_ref_dir))
        print(f"FID: {result['fid']:.4f}")
    else:
        print("FID reference not provided (neither stats nor dir). Skipping FID.")

    # No-reference quality metrics
    print("initializing no-reference metrics (BRISQUE, NIQE, MUSIQ) ...")
    metric_brisque = pyiqa.create_metric("brisque").to(device)
    metric_niqe = pyiqa.create_metric("niqe").to(device)
    metric_musiq = pyiqa.create_metric("musiq").to(device)

    total_images = len(sr_files)
    print(f"processing {total_images} SR images in batches of {batch_size}")

    # 累积平均值
    sum_brisque = 0.0
    sum_niqe = 0.0
    sum_musiq = 0.0
    counted = 0
    failed_batches = 0

    for i in range(0, total_images, batch_size):
        end_idx = min(i + batch_size, total_images)
        batch_files = sr_files[i:end_idx]

        try:
            # 加载图像批次
            batch_imgs = read_images(batch_files)
            
            # 过滤掉空图像
            valid_imgs = []
            valid_files = []
            for j, img in enumerate(batch_imgs):
                if img is not None and img.size > 0:
                    valid_imgs.append(img)
                    valid_files.append(batch_files[j])
                else:
                    print(f"Warning: Skipping invalid image: {batch_files[j]}")
            
            if len(valid_imgs) == 0:
                print(f"Warning: No valid images in batch {i//batch_size + 1}, skipping...")
                failed_batches += 1
                continue

            batch_tensor = optimized_tensor_conversion(valid_imgs, device)

            if batch_tensor.size(0) == 0:
                print(f"Warning: No valid images in batch {i//batch_size + 1}, skipping...")
                failed_batches += 1
                continue

            # 计算批次指标
            brisque_scores = batched_iqa(metric_brisque, batch_tensor, None, desc=f"BRISQUE [{i//batch_size + 1}] ")
            niqe_scores = batched_iqa(metric_niqe, batch_tensor, None, desc=f"NIQE [{i//batch_size + 1}] ")
            musiq_scores = batched_iqa(metric_musiq, batch_tensor, None, desc=f"MUSIQ [{i//batch_size + 1}] ")

            bsz = batch_tensor.size(0)
            sum_brisque += float(brisque_scores.mean().item()) * bsz
            sum_niqe += float(niqe_scores.mean().item()) * bsz
            sum_musiq += float(musiq_scores.mean().item()) * bsz
            counted += bsz

        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            print(f"Batch files: {batch_files}")
            failed_batches += 1
            continue
        finally:
            # 清理内存
            if 'batch_imgs' in locals():
                del batch_imgs
            if 'batch_tensor' in locals():
                del batch_tensor
            if 'brisque_scores' in locals():
                del brisque_scores
            if 'niqe_scores' in locals():
                del niqe_scores
            if 'musiq_scores' in locals():
                del musiq_scores
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 最终平均值
    if counted > 0:
        result["brisque"] = sum_brisque / counted
        result["niqe"] = sum_niqe / counted
        result["musiq"] = sum_musiq / counted
    else:
        result["brisque"] = 0.0
        result["niqe"] = 0.0
        result["musiq"] = 0.0

    result["failed_batches"] = failed_batches
    result["successful_images"] = counted
    result["total_images"] = total_images

    print(f"BRISQUE: {result['brisque']:.4f}")
    print(f"NIQE:    {result['niqe']:.4f}")
    print(f"MUSIQ:   {result['musiq']:.4f}")
    print(f"Successfully processed: {counted}/{total_images} images")
    print(f"Failed batches: {failed_batches}")

    return result


# 其余函数保持不变...
def save_result(out_dir: str, metrics: dict):
    os.makedirs(out_dir, exist_ok=True)
    with open(osp.join(out_dir, "rslt_noref.txt"), "w") as f:
        f.write(json.dumps(metrics, indent=2))


def iter_target_dirs(methods_root: str = None, method_dirs = None):
    """Yield target SR directories to evaluate."""
    candidates = []
    if method_dirs:
        candidates.extend(list(method_dirs))
    if methods_root:
        if not osp.isdir(methods_root):
            raise FileNotFoundError(f"methods_root not found: {methods_root}")
        for name in os.listdir(methods_root):
            p = osp.join(methods_root, name)
            if osp.isdir(p):
                candidates.append(p)

    for c in natsorted(candidates):
        step_dirs = [osp.join(c, d) for d in os.listdir(c) if d.startswith('s') and osp.isdir(osp.join(c, d))] if osp.isdir(c) else []
        if step_dirs:
            for s in natsorted(step_dirs):
                yield s
        else:
            yield c


def extract_method_step(sr_path: str) -> tuple:
    base = osp.basename(sr_path.rstrip('/'))
    parent = osp.basename(osp.dirname(sr_path.rstrip('/')))
    if base.startswith('s') and parent:
        return parent, base
    return base, base


def process_one(sr_path: str, fid_ref_path: str, batch_size: int, save_to: str = None, fid_ref_stats: str = None) -> dict:
    print("==> ", sr_path)
    try:
        metrics = compute_no_ref_metrics(sr_path, fid_ref_path, batch_size, fid_ref_stats=fid_ref_stats)
        out_dir = save_to if save_to else sr_path
        save_result(out_dir, metrics)
        method, step = extract_method_step(sr_path)
        row = {
            "method": method,
            "step": step,
            "sr_dir": sr_path,
            "fid_ref_dir": metrics.get("fid_ref_dir"),
            "fid_ref_stats": metrics.get("fid_ref_stats"),
            "fid": metrics.get("fid"),
            "brisque": metrics.get("brisque"),
            "niqe": metrics.get("niqe"),
            "musiq": metrics.get("musiq"),
            "calculated_at": metrics.get("calculated_at"),
        }
        return row
    except Exception as e:
        print(f"Error processing {sr_path}: {e}")
        return None


if __name__ == "__main__":
    parser = ArgumentParser(description="Compute no-reference metrics: BRISQUE, NIQE, MUSIQ, and FID")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sr_path", type=str, help="Single SR directory to evaluate")
    group.add_argument("--methods_root", type=str, help="Root directory that contains multiple method subfolders")
    group.add_argument("--method_dirs", nargs='+', help="One or more SR method directories to evaluate")
    parser.add_argument("--fid_ref_path", type=str, default=None, help="Reference image directory for FID")
    parser.add_argument("--fid_ref_stats", type=str, default=None, help="Reference stats (.npz) for FID")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for metric computation")
    parser.add_argument("--save_to", type=str, default=None, help="Optional output directory to save results")
    parser.add_argument("--export_excel", type=str, default=None, help="Export Excel summary to this path")
    args = parser.parse_args()

    fid_ref_path = args.fid_ref_path
    fid_ref_stats = args.fid_ref_stats
    batch_size = args.batch_size

    rows = []
    if args.sr_path:
        sr_path = args.sr_path
        if not osp.isdir(sr_path):
            raise FileNotFoundError(f"SR directory not found: {sr_path}")
        row = process_one(sr_path, fid_ref_path, batch_size, save_to=args.save_to, fid_ref_stats=fid_ref_stats)
        if row is not None:
            rows.append(row)
    else:
        targets = list(iter_target_dirs(args.methods_root, args.method_dirs))
        if len(targets) == 0:
            raise RuntimeError("No valid SR directories found to evaluate.")
        for t in targets:
            if not osp.isdir(t):
                print(f"skip (not a dir): {t}")
                continue
            row = process_one(t, fid_ref_path, batch_size, save_to=None, fid_ref_stats=fid_ref_stats)
            if row is not None:
                rows.append(row)

    if args.export_excel and len(rows) > 0:
        out_path = args.export_excel
        os.makedirs(osp.dirname(out_path) or '.', exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_excel(out_path, index=False)
        print(f"Results saved to {out_path}")

    print("Done.")