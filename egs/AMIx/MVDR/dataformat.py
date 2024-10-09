from local.amix_dataset import AMIxDataset
from torch.utils.data import DataLoader
import json
import torch
import soundfile as sf
from tqdm import tqdm

train = json.load(open("data_clips_synthesis_8array1_st4k_a/train.json"))
test = json.load(open("data_clips_synthesis_8array1_st4k_a/test.json"))
val = json.load(open("data_clips_synthesis_8array1_st4k_a/validation.json"))

print(f'The train set contains {len(train)} files.')
print(f'The test set contains {len(test)} files.')
print(f'The validation set contains {len(val)} files.')
# mixtures_shape = {}
# n = 0
# cmp = 0
# for index in tqdm(range(len(train))):
#     # while cmp < 100:
#     c_ex = train[index]
#     # mics = [x for x in c_ex.keys()]
#     mixtures = []
#     sources = []
#     for channels in c_ex:
#         c_mic = c_ex[channels]
#         if "spk1" in c_mic and "spk2" in c_mic and "mixture" in c_mic and "length" in c_mic:
#             cmp += 1
# print("*" * 50)
# for index in tqdm(range(len(val))):
#     # while cmp < 100:
#     c_ex = train[index]
#     mics = [x for x in c_ex.keys()]
#     mixtures = []
#     sources = []
#     if not "1" in c_ex:
#         print(mics)
# print("*" * 50)
# for index in tqdm(range(len(test))):
#     # while cmp < 100:
#     c_ex = train[index]
#     mics = [x for x in c_ex.keys()]
#     mixtures = []
#     sources = []
#     if not "1" in c_ex:
#         print(mics)

# print(cmp)
# length = c_mic["length"]

#     mixture, fs = sf.read(c_mic["mixture"], dtype="float32")  # load all
#     spk1, fs = sf.read(c_mic["spk1"], dtype="float32")
#     spk2, fs = sf.read(c_mic["spk2"], dtype="float32")

#     mixture = torch.from_numpy(mixture).unsqueeze(0)
#     spk1 = torch.from_numpy(spk1).unsqueeze(0)
#     spk2 = torch.from_numpy(spk2).unsqueeze(0)

#     assert fs == 16000
#     mixtures.append(mixture)
#     sources.append(torch.cat((spk1, spk2), 0))

# mixtures = torch.cat(mixtures, 0)
# sources = torch.stack(sources)
# # we pad till max_mic
# valid_mics = mixtures.shape[0]

# if len(mixtures.shape) != 1:
#     mixtures = mixtures[:, :1]
#     mixtures = torch.squeeze(mixtures, dim=1)
#     sources = sources[:, :, :, :1]
#     sources = torch.squeeze(sources, dim=3)

# print(mixtures.shape)
# print(sources.shape)
# # print(valid_mics)
# if sources.shape[1] != 2:
#     print(sources.shape)
# cmp += 1
