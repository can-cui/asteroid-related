import os
import glob
import re
from pathlib import Path
import soundfile as sf
import json
from tqdm import tqdm


def prepare_dataset(meeting_f):
    c_ex_list = []
    for sample_dir in glob.glob(os.path.join(meeting_f, "*")):
        c_ex = {}
        wavs = glob.glob(os.path.join(sample_dir, "*.wav"))
        if len(wavs) == 24:  # check wav number, 8 mixtures, 8 spk1, 8 spk2
            for wav in wavs:
                source_or_mix = Path(wav).stem.split("_")[0]
                n_mic = int(re.findall("\d+", Path(wav).stem.split("_")[-1])[0])
                length = len(sf.SoundFile(wav))

                if n_mic in [1, 2, 3, 4, 5, 6, 7, 8] and n_mic not in c_ex.keys():
                    c_ex[n_mic] = {source_or_mix: wav, "length": length}
                elif n_mic in [1, 2, 3, 4, 5, 6, 7, 8] and n_mic in c_ex.keys():
                    assert c_ex[n_mic]["length"] == length
                    c_ex[n_mic][source_or_mix] = wav
            c_ex_list.append(c_ex)
    return c_ex_list


def parse_dataset(in_dir, out_json_tr, out_json_tt, out_json_cv):
    # All data are divided into train set, test set, and validation set
    # by file name. The division is based on corpus partition of AMI

    examples_tr = []
    examples_tt = []
    examples_cv = []
    floder_list = glob.glob(os.path.join(in_dir, "*"))
    for index in tqdm(range(len(floder_list))):
        meeting_f = floder_list[index]
        folder_name = Path(meeting_f).stem
        # Get meeting id
        meeting_id = folder_name[:7]
        # training part of seen data
        if meeting_id in [
            "ES2002a",
            "ES2003a",
            "ES2005a",
            "ES2006a",
            "ES2007a",
            "ES2008a",
            "ES2009a",
            "ES2010a",
            "ES2012a",
            "ES2013a",
            "ES2014a",
            "ES2015a",
            "ES2016a",
            "IN1001",
            "IN1002",
            "IN1005",
            "IN1007",
            "IN1008",
            "IN1009",
            "IN1012",
            "IN1013",
            "IN1014",
            "IN1016",
            "IS1000a",
            "IS1001a",
            "IS1003a",
            "IS1004a",
            "IS1005a",
            "IS1006a",
            "IS1007a",
            "TS3005a",
            "TS3006a",
            "TS3007a",
            "TS3008a",
            "TS3009a",
            "TS3010a",
            "TS3011a",
            "TS3012a",
            "EN2001a",
            "EN2003a",
            "EN2004a",
            "EN2005a",
            "EN2006a",
            "EN2009a",
        ]:  # all a
            # if meeting_id in [
            #         "ES2002", "ES2003", "ES2005", "ES2006", "ES2007", "ES2008",
            #         "ES2009", "ES2010", "ES2012", "ES2013", "ES2014", "ES2015",
            #         "ES2016", "IS1000", "IS1001", "IS1002", "IS1003", "IS1004",
            #         "IS1005", "IS1006", "IS1007", "TS3005", "TS3006", "TS3007",
            #         "TS3008", "TS3009", "TS3010", "TS3011", "TS3012", "EN2001",
            #         "EN2003", "EN2004", "EN2005", "EN2006", "EN2009", "IN1001",
            #         "IN1002", "IN1005", "IN1007", "IN1008", "IN1009", "IN1012",
            #         "IN1013", "IN1014", "IN1016"
            # ]:  # all data
            c_ex_tr = prepare_dataset(meeting_f)
            examples_tr.extend(c_ex_tr)

        # dev part of seen data
        elif meeting_id in [
            "ES2011a",
            "IS1008a",
            "TS3004a",
            "IB4001",
            "IB4002",
            "IB4003",
            "IB4004",
            "IB4010",
            "IB4011",
        ]:
            # elif meeting_id in [
            #         "ES2011", "IS1008", "TS3004", "IB4001", "IB4002", "IB4003",
            #         "IB4004", "IB4010", "IB4011"
            # ]:
            c_ex_cv = prepare_dataset(meeting_f)
            examples_cv.extend(c_ex_cv)

        # unseen data for evaluation
        elif meeting_id in ["ES2004a", "IS1009a", "TS3003a", "EN2002a"]:
            # elif meeting_id in ["ES2004", "IS1009", "TS3003", "EN2002"]:
            c_ex_tt = prepare_dataset(meeting_f)
            examples_tt.extend(c_ex_tt)

    print(f"The train set contains {len(examples_tr)} files.")
    print(f"The test set contains {len(examples_tt)} files.")
    print(f"The validation set contains {len(examples_cv)} files.")

    with open(out_json_tr, "w") as f:
        json.dump(examples_tr, f, indent=4)
    with open(out_json_tt, "w") as f:
        json.dump(examples_tt, f, indent=4)
    with open(out_json_cv, "w") as f:
        json.dump(examples_cv, f, indent=4)


if __name__ == "__main__":
    in_dir = "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets/AMI/clips_synthesis_8array1_align"
    out_dir = "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/asteroid/egs/AMIx/TAC/data_clips_synthesis_8array1_align_a"
    # in_dir = "/Users/ccui/Desktop/AMI_sample/clips_synthesis_8array1_align"
    # out_dir = "/Users/ccui/Desktop/AMI_sample/data/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_json_tr = os.path.join(out_dir, "train.json")
    out_json_tt = os.path.join(out_dir, "test.json")
    out_json_cv = os.path.join(out_dir, "validation.json")
    parse_dataset(in_dir, out_json_tr, out_json_tt, out_json_cv)
    print(f"Data saved in {out_dir}")
