import os
import librosa
import numpy as np
import warnings
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import cityblock
from concurrent.futures import ProcessPoolExecutor

def compute_snr(ref, target):
    noise = ref - target
    noise_power = np.sum(noise ** 2)
    signal_power = np.sum(ref ** 2)
    if noise_power == 0:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)

def compute_melspec(y, sr=16000, n_mels=80):
    try:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        S_dB = librosa.power_to_db(S, ref=np.max)
        return S_dB.T  # shape: (frames, mels)
    except Exception as e:
        warnings.warn(f"[ERROR] Mel extraction failed: {e}")
        return None

def compute_mel_l1_distance(mel_ref, mel_synth):
    distance, path = fastdtw(mel_ref, mel_synth, dist=cityblock)
    aligned_ref = np.array([mel_ref[i] for i, _ in path])
    aligned_synth = np.array([mel_synth[j] for _, j in path])
    l1_diffs = np.abs(aligned_ref - aligned_synth).mean(axis=1)
    avg_l1 = np.mean(l1_diffs)
    return avg_l1, aligned_ref.shape[0]

def process_folder(folder_name, dir_enh, dir_warped):
    print(f"🔍 Processing folder: {folder_name}")
    dir_ref = os.path.join(dir_enh, folder_name)
    dir_conv = os.path.join(dir_warped, folder_name)

    snrs, mel_diffs = [], []
    total_frames = 0

    if not os.path.isdir(dir_ref) or not os.path.isdir(dir_conv):
        warnings.warn(f"[MISSING] Directory not found: {dir_ref} or {dir_conv}")
        return None

    for file in os.listdir(dir_ref):
        if file.endswith(".wav") and os.path.isfile(os.path.join(dir_conv, file)):
            ref_path = os.path.join(dir_ref, file)
            conv_path = os.path.join(dir_conv, file)
            print(f"[INFO] Comparing: {ref_path} <-> {conv_path}")

            try:
                ref_y, _ = librosa.load(ref_path, sr=16000)
                conv_y, _ = librosa.load(conv_path, sr=16000)
                
                #plt.plot(ref_y, label='Original')
                #plt.plot(conv_y, label='Warped')
                #plt.legend(); plt.title('Waveform Comparison')
                #plt.show()

                if ref_y.shape[0] == 0 or conv_y.shape[0] == 0:
                    warnings.warn(f"[EMPTY] {file} in {folder_name} is empty.")
                    continue

                min_len = min(len(ref_y), len(conv_y))
                ref_y, conv_y = ref_y[:min_len], conv_y[:min_len]

                snr_val = compute_snr(ref_y, conv_y)
                mel_ref = compute_melspec(ref_y)
                mel_conv = compute_melspec(conv_y)

                if mel_ref is None or mel_conv is None:
                    continue

                mel_diff, frames = compute_mel_l1_distance(mel_ref, mel_conv)

                snrs.append(snr_val)
                mel_diffs.append(mel_diff)
                total_frames += frames

            except Exception as e:
                warnings.warn(f"[ERROR] Processing failed for {file} in {folder_name}: {e}")
                continue

    if snrs and mel_diffs:
        mean_snr = np.mean(snrs)
        mean_mel = np.mean(mel_diffs)
        std_mel = np.std(mel_diffs)
        print(f"✅ Folder '{folder_name}': SNR = {mean_snr:.2f} dB, Mel L1 Mean = {mean_mel:.4f}, Std = {std_mel:.4f}, Frames = {total_frames}")
        return mean_snr, mean_mel, std_mel, total_frames
    else:
        warnings.warn(f"[WARNING] No valid metrics computed for folder {folder_name}.")
        return None

def process_all_folders(dir_enh, dir_warped, save_root, folder_names):
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_folder, folder, dir_enh, dir_warped)
            for folder in folder_names
        ]

        snr_vals, mel_means, mel_stds, total_frames_all = [], [], [], 0
        for fut in futures:
            result = fut.result()
            if result:
                snr, mel_mean, mel_std, frames = result
                snr_vals.append(snr)
                mel_means.append(mel_mean)
                mel_stds.append(mel_std)
                total_frames_all += frames

    if snr_vals and mel_means:
        overall_snr = np.mean(snr_vals)
        overall_mel = np.mean(mel_means)
        overall_mel_std = np.mean(mel_stds)

        print(f"\n📊 Overall SNR = {overall_snr:.2f} dB")
        print(f"📊 Overall Mel L1 Mean = {overall_mel:.4f}")
        print(f"📊 Overall STD = {overall_mel_std:.4f}")
        print(f"📊 Total Frames Used = {total_frames_all}")

        result_path = os.path.join(save_root, "snr_mel_summary.txt")
        with open(result_path, "w") as f:
            f.write(f"Overall SNR: {overall_snr:.2f} dB\n")
            f.write(f"Overall Mel L1 Mean: {overall_mel:.4f}\n")
            f.write(f"Overall STD: {overall_mel_std:.4f}\n")
            f.write(f"Total Frames: {total_frames_all}\n")
        print(f"💾 Saved summary to {result_path}")
    else:
        print("[WARNING] No valid results computed.")

# ---------------- MAIN ----------------
if __name__ == '__main__':
    dir1 = "/home/drsandipan/Desktop/VTLN-Experiment/WER_API/VTLN_Experiment_1/Enhanced_Audio/"
    dir2 = "/home/drsandipan/Desktop/VTLN-Experiment/WER_API/VTLN_Experiment_1/Original_Audio/"
    save_root = "/home/drsandipan/Desktop/VTLN-Experiment/WER_API/"
    folder_names = ["3a1f", "4a1a", "5a1a"]

    process_all_folders(dir1, dir2, save_root, folder_names)

