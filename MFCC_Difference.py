import os
import librosa
import numpy as np
import warnings
from fastdtw import fastdtw
from scipy.spatial.distance import cityblock  # L1 distance
from concurrent.futures import ProcessPoolExecutor


def compute_mfcc(file_path, sr=16000, n_mfcc=13):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        if y.shape[0] == 0:
            warnings.warn(f"[EMPTY] File {file_path} is empty.")
            return None
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return mfcc.T  # shape: (frames, n_mfcc)
    except Exception as e:
        warnings.warn(f"[ERROR] Failed MFCC extraction from {file_path}: {e}")
        return None


def compute_mfcc_l1_distance(mfcc_ref, mfcc_synth):
    distance, path = fastdtw(mfcc_ref, mfcc_synth, dist=cityblock)
    aligned_ref = np.array([mfcc_ref[i] for i, _ in path])
    aligned_synth = np.array([mfcc_synth[j] for _, j in path])

    l1_diffs = np.abs(aligned_ref - aligned_synth).mean(axis=1)
    avg_l1 = np.mean(l1_diffs)
    return avg_l1, aligned_ref.shape[0]


def process_folder(folder_name, dir_enh, dir_orig):
    print(f"🔍 Processing folder: {folder_name}")
    dir_ref = os.path.join(dir_orig, folder_name)
    dir_conv = os.path.join(dir_enh, folder_name)

    mfcc_diffs = []
    total_frames = 0

    if not os.path.isdir(dir_ref) or not os.path.isdir(dir_conv):
        warnings.warn(f"[MISSING] Directory not found: {dir_ref} or {dir_conv}")
        return None

    for file in os.listdir(dir_ref):
        if file.endswith(".wav") and os.path.isfile(os.path.join(dir_conv, file)):
            ref_path = os.path.join(dir_ref, file)
            conv_path = os.path.join(dir_conv, file)

            mfcc_ref = compute_mfcc(ref_path)
            mfcc_conv = compute_mfcc(conv_path)

            if mfcc_ref is None or mfcc_conv is None:
                continue

            try:
                diff, frames_used = compute_mfcc_l1_distance(mfcc_ref, mfcc_conv)
                mfcc_diffs.append(diff)
                total_frames += frames_used
            except Exception as e:
                warnings.warn(f"[ERROR] DTW failed for {file} in {folder_name}: {e}")

    if mfcc_diffs:
        mean_diff = np.mean(mfcc_diffs)
        std_diff = np.std(mfcc_diffs)
        print(f"✅ Folder '{folder_name}': MFCC L1 Mean = {mean_diff:.4f}, Std = {std_diff:.4f}, Frames = {total_frames}")
        return mean_diff, std_diff, total_frames
    else:
        warnings.warn(f"[WARNING] No valid MFCC diff computed for folder {folder_name}.")
        return None


def process_all_folders(dir_enh, dir_orig, save_root, folder_names):
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_folder, folder, dir_enh, dir_orig)
            for folder in folder_names
        ]

        mfcc_means, mfcc_stds, total_frames_all = [], [], 0
        for fut in futures:
            result = fut.result()
            if result:
                mfcc_mean, mfcc_std, frames = result
                mfcc_means.append(mfcc_mean)
                mfcc_stds.append(mfcc_std)
                total_frames_all += frames

    if mfcc_means:
        overall_mean = np.mean(mfcc_means)
        overall_std = np.mean(mfcc_stds)

        print(f"\n📊 Overall Average MFCC L1 Distance = {overall_mean:.4f}")
        print(f"📊 Overall STD = {overall_std:.4f}")
        print(f"📊 Total Frames Used = {total_frames_all}")

        # Save result
        result_path = os.path.join(save_root, "mfcc_diff_summary.txt")
        with open(result_path, "w") as f:
            f.write(f"Overall MFCC L1 Mean: {overall_mean:.4f}\n")
            f.write(f"Overall STD: {overall_std:.4f}\n")
            f.write(f"Total Frames: {total_frames_all}\n")
        print(f"💾 Saved MFCC summary to {result_path}")
    else:
        print("[WARNING] No valid MFCC differences computed.")


# ---------------- MAIN ----------------
if __name__ == '__main__':
    dir1 = "/home/drsandipan/Desktop/VTLN-Experiment/WER_API/VTLN_Experiment_1/Enhanced_Audio/"
    dir2 = "/home/drsandipan/Desktop/VTLN-Experiment/VTLN_Experiment_1/Original_Audio/"
    save_root = "/home/drsandipan/Desktop/VTLN-Experiment/WER_API/"
    folder_names = ["3a1f", "4a1a", "5a1a"]

    process_all_folders(dir1, dir2, save_root, folder_names)

