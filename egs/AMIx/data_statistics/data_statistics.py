import glob
import os
import json
import xml.etree.ElementTree as et
import soundfile as sf
import numpy as np
from pathlib import Path
import itertools
from tqdm import tqdm
from collections import Counter


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


# Get respectively the list of length statistics for 1, 2, 3, and 4 people talking at the same
# time based on all annotations and reference audio
def get_overlap_statistics(seg_path_list, ref_audio_path):
    ref_audio, fs = sf.read(
        ref_audio_path,
        dtype="float32",
    )
    total_dur = int(len(ref_audio) / fs * 1000)
    # Add up the binary list of all sources, if the value of a position is 1, 2, 3 or 4, then
    # there are 1, 2, 3 and 4 people talking at this moment respectively
    for index in range(len(seg_path_list)):
        seg_path = seg_path_list[index]
        seg_list = get_binary_list(seg_path, total_dur)
        if index == 0:
            seg_overlap = seg_list
        else:
            seg_overlap += seg_list
    count = Counter(seg_overlap)
    # ovlSpk_num_list = [1, 2, 3, 4, 5]
    # overlap_list = []
    # # For each type of overlap, count the duration of each overlap and generate a list
    # for ovlSpk_num in ovlSpk_num_list:
    #     overlap = [
    #         len(list(v)) for k, v in itertools.groupby(seg_overlap)
    #         if k == ovlSpk_num
    #     ]
    #     overlap_list.append(overlap)
    return count


def main():
    # audio_path = "/Users/ccui/Desktop/AMI_sample/amicorpus"
    # segment_path = "/Users/ccui/Desktop/AMI_sample/annotations/segments"
    audio_path = "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets/AMI/amicorpus"
    segment_path = "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets/AMI/annotations/segments"
    folder_list = glob.glob(os.path.join(audio_path, "*"))
    segment_list = glob.glob(os.path.join(segment_path, "*"))
    for index in tqdm(range(len(folder_list))):
        folder = folder_list[index]
        meeting_name = Path(folder).stem.split(".")[-1]
        # Use Headset0 of each session as reference audio
        ref_audio_path = os.path.join(audio_path, meeting_name, "audio",
                                      meeting_name + ".Headset-0.wav")
        seg_path_list = []
        for segment in segment_list:
            seg_meeting_name = Path(segment).stem.split(".")[0]
            if seg_meeting_name == meeting_name:
                seg_path_list.append(str(Path(segment)))
        # overlap_list,count = get_overlap_statistics(seg_path_list, ref_audio_path)
        # # Aggregate all meeting data into one list
        # if index == 0:
        #     overlap_all_num = overlap_list
        # else:
        #     for idx in range(len(overlap_list)):
        #         overlap = overlap_list[idx]
        #         overlap_all_num[idx].extend(overlap)
        count = get_overlap_statistics(seg_path_list, ref_audio_path)
        if index == 0:
            count_all = count
        else:
            count_all += count

    # with open('statistics_5.json', "w") as f:
    #     json.dump(overlap_all_num, f, indent=4)
    with open('count_overlap.json', "w") as f:
        json.dump(count_all, f, indent=4)


if __name__ == "__main__":
    main()