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
                             meeting_folder, array_path_list, fs, path_type):
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
                path_type, array_path.split(".")[-2][-9:])
            if not os.path.exists(file_name):
                sf.write(
                    file_name,
                    clip,
                    fs,
                )
            # clips_list.append(file_name)
    # return clips_list


# Get the agent label from the number of headset
def get_annotation_spk(map_chan2agent_xml, meeting_name, headset_id):
    tree = et.parse(map_chan2agent_xml)
    root = tree.getroot()
    for meeting in root.findall("meeting"):
        if meeting.get('observation') == meeting_name:
            for speaker in meeting.findall("speaker"):
                if speaker.get('channel') == headset_id:
                    annotation_spk = speaker.get('nxt_agent')
    return annotation_spk


def main(job_nb):
    # in_dir = "/Users/ccui/Desktop/AMI_sample/amicorpus_4headsets_align"
    in_dir = "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets/AMI/amicorpus_4headsets_align"
    folder_list = glob.glob(os.path.join(in_dir, "*"))
    # annotation_dir = "/Users/ccui/Desktop/AMI_sample/annotations"
    annotation_dir = "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets/AMI/annotations"
    segment_list = glob.glob(os.path.join(annotation_dir, "segments/*"))
    map_chan2agent_xml = os.path.join(annotation_dir,
                                      "corpusResources/meetings.xml")
    # local_save_dir = "/Users/ccui/Desktop/AMI_sample/clips_monoSpk_8array1_align"
    local_save_dir = "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets/AMI/clips_monoSpk_8array1_align"
    if not os.path.exists(local_save_dir):
        os.makedirs(local_save_dir)
    win_size = 4000  # 4 seconds
    thresh = 4000
    step = 4000
    # Get a list of 8 array1's meeting
    folder_list_8array1 = []

    for folder in folder_list:
        # wavs = glob.glob(os.path.join(folder, "audio/*"))
        wavs = glob.glob(os.path.join(folder, "*"))
        cpt = 0
        for wav in wavs:
            mic_type = Path(wav).stem.split(".")[-1].split("-")[0]
            if mic_type == "Array1":
                cpt += 1
        if cpt == 8:
            folder_list_8array1.append(folder)

    folder = folder_list_8array1[job_nb]
    # wavs = glob.glob(os.path.join(folder, "audio/*"))
    wavs = glob.glob(os.path.join(folder, "*"))
    meeting_name = Path(folder).stem.split(".")[-1]
    meeting_folder = os.path.join(local_save_dir, meeting_name)
    if not os.path.exists(meeting_folder):
        os.makedirs(meeting_folder)
    # Get all array channels of this meeting
    array_path_list = []
    source_path_dict = {}
    for wav in wavs:
        mic_type = Path(wav).stem.split(".")[-1].split("-")[0]
        if mic_type == "Array1":
            array_path_list.append(str(Path(wav)))
        elif mic_type == "Source" and Path(wav).stem.split(".")[-1].split(
                "_")[-1].split("-")[0] == "Array1":
            headset_num = Path(wav).stem.split(".")[-1].split("-")[-2][0]
            if not headset_num in source_path_dict:
                source_path_dict[headset_num] = [str(Path(wav))]
            else:
                source_path_dict[headset_num].append(str(Path(wav)))
    # Get all seg annotations of this meeting
    seg_path_list = []
    for seg in segment_list:
        seg_meeting_name = Path(seg).stem.split(".")[0]
        if seg_meeting_name == meeting_name:
            seg_path_list.append(str(Path(seg)))
    # Get reference audio duration
    array_ref, fs = sf.read(
        array_path_list[0],
        dtype="float32",
    )
    total_dur = int(len(array_ref) / fs * 1000)
    for target_spk_hds in source_path_dict.keys():
        source_path_list = source_path_dict[target_spk_hds]
        target_spk = get_annotation_spk(map_chan2agent_xml, meeting_name,
                                        target_spk_hds)
        target = get_target_from_targetSpk(target_spk, total_dur,
                                           seg_path_list)
        # Statistics for each duration
        seg_dur = [
            len(list(v)) for k, v in itertools.groupby(target) if k == 1
        ]
        # Operates only when the maximum duration exceeds or equals the threshold
        if max(seg_dur) >= thresh:
            get_clips_from_targetSpk(win_size,
                                     thresh,
                                     step,
                                     target,
                                     target_spk,
                                     meeting_folder,
                                     array_path_list,
                                     fs,
                                     path_type="Array")
            get_clips_from_targetSpk(win_size,
                                     thresh,
                                     step,
                                     target,
                                     target_spk,
                                     meeting_folder,
                                     source_path_list,
                                     fs,
                                     path_type="Source")


if __name__ == "__main__":
    # One meeting per job
    args = parser.parse_args()
    job_nb = int(args.job_nb) - 1
    main(job_nb)