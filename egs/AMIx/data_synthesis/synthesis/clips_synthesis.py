import glob
import os
import soundfile as sf
from pathlib import Path
import argparse
import itertools

parser = argparse.ArgumentParser(
    description='Pass a number to the command line')
parser.add_argument('job_nb', type=str, help='job number')


def main(job_nb):
    # in_dir = "/Users/ccui/Desktop/AMI_sample/clips_monoSpk"
    # out_dir = "/Users/ccui/Desktop/AMI_sample/clips_synthesis"

    in_dir = "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets/AMI/clips_monoSpk_8array1_st4k"
    out_dir = "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets/AMI/clips_synthesis_8array1_st4k"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    folder_list = glob.glob(os.path.join(in_dir, "*"))
    # for meeting in folder_list:
    meeting = folder_list[job_nb]
    meeting_name = Path(meeting).stem.split(".")[-1]
    spk_list = glob.glob(os.path.join(meeting, "*"))
    # Combine speakers in pairs
    spk_comb = list(itertools.combinations(spk_list, 2))
    sample = 1
    for couple in spk_comb:
        spk1 = couple[0]
        spk2 = couple[1]
        # Get all samples for each speaker
        spk1_samples = glob.glob(os.path.join(spk1, "*"))
        spk2_samples = glob.glob(os.path.join(spk2, "*"))
        # Combine samples in pairs
        couple_com_list = list(itertools.product(spk1_samples, spk2_samples))
        for couple_com in couple_com_list:
            # Each sample has multiple arrays
            arrays_spk1 = glob.glob(os.path.join(couple_com[0], "*"))
            arrays_spk2 = glob.glob(os.path.join(couple_com[1], "*"))
            mic_num = 1
            # Synthesis two different sets of speakers with the same array number
            for arr_spk1 in arrays_spk1:
                arr_num_spk1 = Path(arr_spk1).stem.split(".")[-1].split(
                    "_")[-1]
                for arr_spk2 in arrays_spk2:
                    arr_num_spk2 = Path(arr_spk2).stem.split(".")[-1].split(
                        "_")[-1]
                    if arr_num_spk1 == arr_num_spk2:
                        mic = "mic" + str(mic_num)
                        sample_folder = os.path.join(out_dir, meeting_name,
                                                     "sample" + str(sample))
                        if not os.path.exists(sample_folder):
                            os.makedirs(sample_folder)
                        spk1_file_name = os.path.join(sample_folder,
                                                      "spk1_" + mic + ".wav")
                        cmd_cp_spk1 = f"cp -f {arr_spk1} {spk1_file_name}"
                        os.system(cmd_cp_spk1)
                        spk2_file_name = os.path.join(sample_folder,
                                                      "spk2_" + mic + ".wav")
                        cmd_cp_spk2 = f"cp -f {arr_spk2} {spk2_file_name}"
                        os.system(cmd_cp_spk2)
                        mixture_file_name = os.path.join(
                            sample_folder, "mixture_" + mic + ".wav")

                        spk1_wav, fs = sf.read(
                            spk1_file_name,
                            dtype="float32",
                        )
                        spk2_wav, fs = sf.read(
                            spk2_file_name,
                            dtype="float32",
                        )
                        sf.write(
                            mixture_file_name,
                            spk1_wav + spk2_wav,
                            fs,
                        )
                        mic_num += 1
            sample += 1


if __name__ == "__main__":
    # One meeting per job
    args = parser.parse_args()
    job_nb = int(args.job_nb) - 1
    main(job_nb)