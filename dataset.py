from torch.utils import data
import pandas as pd
import numpy as np

CHARSET = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, \
           'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, \
           'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20, \
           'O': 20, 'U': 20,
           'B': (2, 11),
           'Z': (3, 13),
           'J': (7, 9)}
CHARLEN = 21

def encoding_seq_np(seq, arr):
    for i, c in enumerate(seq):
        if c == "_" or c == "*":
            # let them zero
            continue
        elif isinstance(CHARSET[c], int):
            idx = CHARSET[c]
            arr[i][idx] = 1
        else:
            idx1 = CHARSET[c][0]
            idx2 = CHARSET[c][1]
            arr[i][idx1] = 0.5
            arr[i][idx2] = 0.5

class PepseqDataset(data.Dataset):
    def __init__(self, file_path, type='train', seq_len = 1000):
        self.type = type
        self.file = file_path
        self.seq_len = 1000
        # column 0 is label, column 1 is seq
        df = pd.read_csv(self.file, sep='\t', header = None)
        self.labels = df[0]
        self.seqs = df[1]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        seq_np = np.zeros((self.seq_len, CHARLEN), dtype=np.float32)
        encoding_seq_np(self.seqs[index], seq_np)
        return seq_np, self.labels[index]