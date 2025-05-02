import os
import requests
from jiwer import wer
from natsort import natsorted

# Constants
API_URL = 'http://localhost:3000/get_asr'
BASE_AUDIO_URL = 'https://s3.ap-south-1.amazonaws.com/speech-evaluator-audio-recordings/speaker-anonymization/VTLN_Output/warp_factor_1.26/'
LOCAL_BASE_PATH = '/home/drsandipan/Desktop/VTLN-Experiment/WER_API/VTLN_Experiment_1/VTLN_Output/warp_factor_1.26/'
OUTPUT_WER_FILE = 'wer_results_speakerwise.txt'
REF_TEXT_PATH = '/home/drsandipan/Desktop/VTLN-Experiment/WER_API/VTLN_Experiment_1/Transcripts/'
FOLDERS = ['3a1f', '4a1a', '5a1a']  # Use only the folders you want here

def compute_wer(reference, hypothesis):
    return wer(reference, hypothesis)

def read_reference_text(ref_id):
    txt_file_path = os.path.join(REF_TEXT_PATH, ref_id + '.txt')
    if not os.path.exists(txt_file_path):
        print(f"[Missing] Reference text not found: {txt_file_path}")
        return None
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def call_api(s3_url, reference_text_id):
    print(f"s3_url is :: {s3_url}")
    headers = {
        'Content-Type': 'application/json',
        'x-api-key': 'rKIXbKZvNA38cByxV8TFG9EOCfkELIfx5wi1UbBV'
    }
    data = {
        's3_url': s3_url,
        'reference_text_id': reference_text_id
    }
    try:
        response = requests.post(API_URL, json=data, headers=headers)
        response_data = response.json().get('response', {})
        if not response_data or "errorMessage" in response_data:
            print(f"[API Error] for {s3_url}")
            return ""
        return response_data.get("decoded_text", "")
    except Exception as e:
        print(f"[Exception] API call failed for {s3_url}: {e}")
        return ""

# Global counters
global_total_wer = 0.0
global_num_samples = 0

with open(OUTPUT_WER_FILE, 'w', encoding='utf-8') as out_file:
    for folder in FOLDERS:
        folder_path = os.path.join(LOCAL_BASE_PATH, folder)
        if not os.path.isdir(folder_path):
            print(f"[Error] Folder not found: {folder_path}")
            continue

        wav_files = natsorted([f for f in os.listdir(folder_path) if f.endswith('.wav')])
        print(f"\n[{folder}] Found {len(wav_files)} wav files")
        out_file.write(f"\n[{folder}]\n")

        # Speaker-wise counters
        speaker_total_wer = 0.0
        speaker_num_samples = 0

        for wav_file in wav_files:
            s3_url = f"{BASE_AUDIO_URL}{folder}/{wav_file}"
            base_name = os.path.splitext(wav_file)[0]

            if base_name.startswith(folder + "_"):
                base_name = base_name[len(folder) + 1:]
            if base_name.endswith('_'):
                base_name = base_name[:-1]

            ref_id = base_name.replace('_', '-')
            print(f"[Info] ref_id: {ref_id}")

            decoded_text = call_api(s3_url, ref_id)
            if not decoded_text:
                out_file.write(f"{folder}/{wav_file}\tAPI failed\n")
                continue

            reference_text = read_reference_text(ref_id)
            if reference_text is None:
                out_file.write(f"{folder}/{wav_file}\tMissing reference text\n")
                continue

            error_rate = compute_wer(reference_text, decoded_text)
            speaker_total_wer += error_rate
            speaker_num_samples += 1
            global_total_wer += error_rate
            global_num_samples += 1

            result_line = f"{folder}/{wav_file}\tWER: {error_rate:.4f}\n"
            print(result_line.strip())
            out_file.write(result_line)

        # Write speaker-wise average WER
        if speaker_num_samples > 0:
            avg_wer = speaker_total_wer / speaker_num_samples
            out_file.write(f"Average WER for speaker {folder}: {avg_wer:.4f}\n")
            print(f"✅ Average WER for {folder}: {avg_wer:.4f}")
        else:
            out_file.write(f"No valid samples for speaker {folder}\n")
            print(f"⚠️ No valid samples for {folder}")

# Global average WER
if global_num_samples > 0:
    overall_avg = global_total_wer / global_num_samples
    print(f"\n✅ Overall Average WER over {global_num_samples} samples: {overall_avg:.4f}")
    with open(OUTPUT_WER_FILE, 'a', encoding='utf-8') as out_file:
        out_file.write(f"\nOverall Average WER: {overall_avg:.4f}\n")
else:
    print("\n⚠️ No valid samples processed.")

