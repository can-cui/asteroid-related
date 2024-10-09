import json
import glob
import os
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description='test')

parser.add_argument('--jsondir',
                    type=str,
                    help='Path to the original json folder')
parser.add_argument('--target_jsondir',
                    type=str,
                    help='Path to the new json folder to be generated')
parser.add_argument('--datadir',
                    type=str,
                    help='Path to the original data folder')
parser.add_argument('--target_datadir',
                    type=str,
                    help='Path to the new data folder')


def main():
    args = parser.parse_args()
    jsondir = args.jsondir
    target_jsondir = args.target_jsondir
    datadir = args.datadir
    target_datadir = args.target_datadir
    json_list = glob.glob(os.path.join(jsondir, "*"))

    for json_f in json_list:
        datatype = Path(json_f).stem
        with open(json_f, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        for clips in dataset:
            for channel_num in clips:
                channel = clips[channel_num]
                for mic_type in channel:
                    # print(mic_type)
                    if mic_type != "length":
                        channel[mic_type] = channel[mic_type].replace(
                            datadir, target_datadir)

        save_path = os.path.join(target_jsondir, datatype + ".json")
        with open(save_path, 'w') as f:
            json.dump(dataset, f)


if __name__ == "__main__":
    main()