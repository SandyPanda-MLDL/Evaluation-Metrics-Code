import os
import numpy as np
import pyworld as pw
import pysptk
import soundfile as sf
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import math
from concurrent.futures import ProcessPoolExecutor
import warnings


def extract_mcep(audio_path, mcep_dim=24, alpha=0.65):
    try:
        x, fs = sf.read(audio_path)
        if x.shape[0] == 0:
            warnings.warn(f"[EMPTY] File {audio_path} is empty.")
            return None

        _f0, t = pw.harvest(x, fs)
        sp = pw.cheaptrick(x, _f0, t, fs)
        mcep = pysptk.sptk.mcep(sp, order=mcep_dim - 1, alpha=alpha)

        if np.any(np.isnan(mcep)) or np.allclose(mcep, 0):
            warnings.warn(f"[WARNING] Invalid or zero MCEP features in {audio_path}")
            return None

        return mcep
    except Exception as e:
        warnings.warn(f"[ERROR] Could not process {audio_path}: {str(e)}")
        return None


def calculate_mcd(mcep_ref, mcep_synth):
    distance, path = fastdtw(mcep_ref, mcep_synth, dist=euclidean)
    aligned_ref = np.array([mcep_ref[i] for i, _ in path])
    aligned_synth = np.array([mcep_synth[j] for _, j in path])

    log_spec_dB_const = 10.0 / math.log(10.0) * math.sqrt(2.0)
    total_dist = 0.0
    for i in range(aligned_ref.shape[0]):
        diff = aligned_ref[i] - aligned_synth[i]
        sq_diff = np.sum(diff ** 2)
        total_dist += sq_diff
        print(f"Frame {i}: Squared diff = {sq_diff:.4f}")

    mcd = log_spec_dB_const * math.sqrt(total_dist / (aligned_ref.shape[0] * aligned_ref.shape[1]))
    return mcd, aligned_ref.shape[0]


def process_folder(folder_name, dir_enh, dir_orig):
    print(f"🔍 Processing folder: {folder_name}")
    dir_ref = os.path.join(dir_orig, folder_name)
    dir_conv = os.path.join(dir_enh, folder_name)

    mcd_values = []
    total_frames = 0

    if not os.path.isdir(dir_ref) or not os.path.isdir(dir_conv):
        warnings.warn(f"[MISSING] Directory not found: {dir_ref} or {dir_conv}")
        return None

    for file in os.listdir(dir_ref):
        if file.endswith(".wav") and os.path.isfile(os.path.join(dir_conv, file)):
            ref_path = os.path.join(dir_ref, file)
            conv_path = os.path.join(dir_conv, file)

            mcep_ref = extract_mcep(ref_path)
            mcep_conv = extract_mcep(conv_path)

            if mcep_ref is None or mcep_conv is None:
                continue

            try:
                mcd, frames_used = calculate_mcd(mcep_ref, mcep_conv)
                mcd_values.append(mcd)
                total_frames += frames_used
            except Exception as e:
                warnings.warn(f"[ERROR] MCD failed for {file} in {folder_name}: {e}")

    if mcd_values:
        mcd_mean = np.mean(mcd_values)
        mcd_std = np.std(mcd_values)
        print(f"✅ Folder '{folder_name}': MCD Mean = {mcd_mean:.4f} dB, MCD Std = {mcd_std:.4f} dB, Frames = {total_frames}")
        return mcd_mean, mcd_std, total_frames
    else:
        warnings.warn(f"[WARNING] No valid MCD calculated for folder {folder_name}.")
        return None


def process_all_folders(dir_enh, dir_orig, save_root, folder_names):
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_folder, folder, dir_enh, dir_orig)
            for folder in folder_names
        ]

        mcd_means, mcd_stds, total_frames_all = [], [], 0
        for fut in futures:
            result = fut.result()
            if result:
                mcd_mean, mcd_std, frames = result
                mcd_means.append(mcd_mean)
                mcd_stds.append(mcd_std)
                total_frames_all += frames

    if mcd_means:
        overall_mcd_mean = np.mean(mcd_means)
        overall_mcd_std = np.mean(mcd_stds)

        print(f"\n📊 Overall Average MCD = {overall_mcd_mean:.4f} dB")
        print(f"📊 Overall Average STD = {overall_mcd_std:.4f} dB")
        print(f"📊 Total Frames Used = {total_frames_all}")

        # Optionally save results
        result_path = os.path.join(save_root, "mcd_summary.txt")
        with open(result_path, "w") as f:
            f.write(f"Overall MCD Mean: {overall_mcd_mean:.4f} dB\n")
            f.write(f"Overall STD: {overall_mcd_std:.4f} dB\n")
            f.write(f"Total Frames: {total_frames_all}\n")
        print(f"💾 Saved MCD summary to {result_path}")
    else:
        print("[WARNING] No valid MCD scores computed from any folder.")


# ---------------- MAIN ----------------
if __name__ == '__main__':
    dir1 = "/home/drsandipan/Desktop/VTLN-Experiment/WER_API/VTLN_Experiment_1/Enhanced_Audio/"
    dir2 = "/home/drsandipan/Desktop/VTLN-Experiment/VTLN_Experiment_1/Original_Audio/"
    save_root = "/home/drsandipan/Desktop/VTLN-Experiment/WER_API/"
    folder_names = ["3a1f", "4a1a", "5a1a"]

    process_all_folders(dir1, dir2, save_root, folder_names)

