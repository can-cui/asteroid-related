import glob
import os
import json
import xml.etree.ElementTree as et
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import itertools

parser = argparse.ArgumentParser(
    description='Pass a number to the command line')
parser.add_argument('job_nb', type=str, help='job number')


# Get an Array of all 0's and 1's based on annotation and the total audio length
def get_binary_list(segment_path, total_dur):
    tree = et.parse(segment_path)
    root = tree.getroot()
    # Create an array of all zeros and length total_dur
    segs_arr = np.zeros(total_dur)
    # convert second string to milliseccond int
    segs = [[
        elem.attrib["transcriber_start"],
        elem.attrib["transcriber_end"],
    ] for elem in root.iter("segment")]
    for dur in segs:
        dur[0] = int(float(dur[0]) * 1000)
        dur[1] = int(float(dur[1]) * 1000)
    # Set the value on the array corresponding to the position of each interval in segs to 1
    for seg in segs:
        start_seg = seg[0]
        end_seg = seg[1] + 1
        segs_arr[start_seg:end_seg] = 1
    return segs_arr


# Get the binary array which indicate the solo part of a target speaker
def get_target_from_targetSpk(target_spk, total_dur, seg_path_list):
    ref_overlap = np.zeros(total_dur)
    for index in range(len(seg_path_list)):
        seg_path = seg_path_list[index]
        spk = seg_path.split(".")[-3]
        if spk == target_spk:
            target_annot = get_binary_list(seg_path, total_dur)
        # Add up all binary array elements that are not the target
        else:
            ref_annot = seg_path
            seg_list = get_binary_list(ref_annot, total_dur)
            ref_overlap += seg_list
    # Subtract the reference array from the target array
    subtr = target_annot - ref_overlap
    # Only keep the value above 0, i.e., 1, which is the solo time of the target
    target = np.maximum(subtr, 0)
    return target


def get_clips_from_targetSpk(win_size, thresh, step, target, target_spk,
                             meeting_folder, array_path_list, fs):
    # clips_list = []
    target_spk_folder = os.path.join(meeting_folder, target_spk)
    if not os.path.exists(target_spk_folder):
        os.makedirs(target_spk_folder)
    start_point_list = list(range(0, len(target) - win_size, step))
    clips_seg_list = []
    for start_point in start_point_list:
        end_point = start_point + win_size
        seg = target[start_point:end_point]
        if seg.sum() == thresh:
            clips_seg_list.append([start_point, end_point])
    for index in range(len(clips_seg_list)):
        clips_seg = clips_seg_list[index]
        start_clip = int(clips_seg[0] * (fs / 1000))
        end_clip = int(clips_seg[1] * (fs / 1000))
        sample_folder = os.path.join(target_spk_folder,
                                     "sample" + str(index + 1))
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)
        for array_path in array_path_list:
            array, fs = sf.read(
                array_path,
                dtype="float32",
            )
            # Some audio (ES2010d) is having two identical channels, remove one
            if len(array.shape) != 1:
                array = array[:, :1]
                array = np.squeeze(array, axis=1)
            clip = array[start_clip:end_clip]
            file_name = sample_folder + "/%s_%s.wav" % (
                target_spk, array_path.split(".")[-2])
            if not os.path.exists(file_name):
                sf.write(
                    file_name,
                    clip,
                    fs,
                )
            # clips_list.append(file_name)
    # return clips_list


def main(job_nb):
    # in_dir = "/Users/ccui/Desktop/AMI_sample/amicorpus"
    in_dir = "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets/AMI/amicorpus"
    folder_list = glob.glob(os.path.join(in_dir, "*"))
    # segment_path = "/Users/ccui/Desktop/AMI_sample/annotations/segments"
    segment_path = "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets/AMI/annotations/segments"
    segment_list = glob.glob(os.path.join(segment_path, "*"))
    # local_save_dir = "/Users/ccui/Desktop/AMI_sample/clips_monoSpk"
    local_save_dir = "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets/AMI/clips_monoSpk_8array1_st4k"
    if not os.path.exists(local_save_dir):
        os.makedirs(local_save_dir)
    win_size = 4000  # 4 seconds
    thresh = 4000
    step = 4000
    # clips = {}
    # clips_num = 0
    # for index in tqdm(range(len(folder_list))):
    # Get a list of 8 array1's meeting
    folder_list_8array1 = []
    for folder in folder_list:
        wavs = glob.glob(os.path.join(folder, "audio/*"))
        cpt = 0
        for wav in wavs:
            mic_type = Path(wav).stem.split(".")[-1].split("-")[0]
            if mic_type == "Array1":
                cpt += 1
        if cpt == 8:
            folder_list_8array1.append(folder)

    folder = folder_list_8array1[job_nb]
    wavs = glob.glob(os.path.join(folder, "audio/*"))
    meeting_name = Path(folder).stem.split(".")[-1]
    # clips[meeting_name] = {}
    # if meeting_name in ["EN2009b","EN2007d","ES2011d","IB4011",]
    meeting_folder = os.path.join(local_save_dir, meeting_name)
    if not os.path.exists(meeting_folder):
        os.makedirs(meeting_folder)
    # Get all array channels of this meeting
    array_path_list = []
    for wav in wavs:
        mic_type = Path(wav).stem.split(".")[-1].split("-")[0]
        if mic_type == "Array1":
            array_path_list.append(str(Path(wav)))
    # Get all seg annotations of this meeting
    seg_path_list = []
    spk_list = []
    for seg in segment_list:
        seg_meeting_name = Path(seg).stem.split(".")[0]
        if seg_meeting_name == meeting_name:
            seg_path_list.append(str(Path(seg)))
            spk = seg.split(".")[-3]
            spk_list.append(spk)
    # Get reference audio duration
    array_ref, fs = sf.read(
        array_path_list[0],
        dtype="float32",
    )
    total_dur = int(len(array_ref) / fs * 1000)
    for target_spk in spk_list:
        # clips[meeting_name][target_spk] = []
        target = get_target_from_targetSpk(target_spk, total_dur,
                                           seg_path_list)
        # Statistics for each duration
        seg_dur = [
            len(list(v)) for k, v in itertools.groupby(target) if k == 1
        ]
        # Operates only when the maximum duration exceeds or equals the threshold
        if max(seg_dur) >= thresh:
            get_clips_from_targetSpk(win_size, thresh, step, target,
                                     target_spk, meeting_folder,
                                     array_path_list, fs)
            # clips_list = get_clips_from_targetSpk(win_size, thresh, step, target,
            #                                       target_spk, meeting_folder,
            #                                       array_path_list, fs)
            # clips[meeting_name][target_spk].append(clips_list)
            # clips_num += len(clips_list)

    # with open('clips.json', "w") as f:
    #     json.dump(clips, f, indent=4)
    # print(f"A total of {clips_num} clips were generated.")


if __name__ == "__main__":
    # One meeting per job
    args = parser.parse_args()
    job_nb = int(args.job_nb) - 1
    main(job_nb)