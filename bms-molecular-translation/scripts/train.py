import re, math
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
import timm
from typing import List
from functools import partial

# local package
from kkutils.util.com import get_args, get_file_list, correct_dirpath, set_logger, save_pickle, load_pickle
from kkutils.lib.nn.module import TorchNN, Layer
from kkutils.lib.nn.model import BaseNN
from kkutils.lib.nn.model_image import ImageDataset, ImageNN, calc_normalize_mean_std
from kkutils.util.image.utils import pil2cv, cv2pil, gray_to_rgb
from kkutils.util.image.transform import ResizeFixRatio
logger = set_logger(__name__)


DIR_IN  = correct_dirpath("../input/")
IMAGE_IN_SIZE = 260 
SEQ_LENGTH    = 256


def split(text: str):
    listwk = [
        '(', ')', '+', ',', '-', '/b', '/c', '/h', '/i', '/m', '/s', '/t',
        'Br', 'B', 'Cl', 'C', 'D', 'F', 'H', 'I', 'N', 'O', 'P', 'Si', 'S', 'T'
    ]
    for x in listwk:
        text = text.replace(x, f" {x} ")
    text = re.sub("\s+", " ", text)
    text = text.replace("B r", "Br").replace("C l", "Cl").replace("S i", "Si").strip()
    return text


class Tokenizer(object):
    
    def __init__(self, max_len: int=512):
        self.stoi = {}
        self.itos = {}
        self.max_len = max_len

    def __len__(self):
        return len(self.stoi)
    
    def fit_on_texts(self, texts):
        vocab = set()
        for text in texts:
            vocab.update(text.split(' '))
        vocab = sorted(vocab)
        vocab.append('<sos>')
        vocab.append('<eos>')
        vocab.append('<pad>')
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = {item[1]: item[0] for item in self.stoi.items()}
        
    def text_to_sequence(self, text):
        sequence = []
        sequence.append(self.stoi['<sos>'])
        text = split(text)
        for s in text.split(' '):
            sequence.append(self.stoi[s])
        sequence.append(self.stoi['<eos>'])
        while(True):
            if len(sequence) >= self.max_len: break
            sequence.append(self.stoi['<pad>'])
        return sequence
    
    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = self.text_to_sequence(text)
            sequences.append(sequence)
        return sequences

    def sequence_to_text(self, sequence):
        return ''.join(list(map(lambda i: self.itos[i], sequence)))
    
    def sequences_to_texts(self, sequences):
        texts = []
        for sequence in sequences:
            text = self.sequence_to_text(sequence)
            texts.append(text)
        return texts
    
    def predict_caption(self, sequence):
        caption = ''
        for i in sequence:
            if i == self.stoi['<eos>'] or i == self.stoi['<pad>']:
                break
            caption += self.itos[i]
        return caption
    
    def predict_captions(self, sequences):
        captions = []
        for sequence in sequences:
            caption = self.predict_caption(sequence)
            captions.append(caption)
        return captions


tokenizer = Tokenizer(max_len=SEQ_LENGTH)
tokenizer.stoi = {'(': 0, ')': 1, '+': 2, ',': 3, '-': 4, '/b': 5, '/c': 6, '/h': 7, '/i': 8, '/m': 9, '/s': 10, '/t': 11, '0': 12, '1': 13, '10': 14, '100': 15, '101': 16, '102': 17, '103': 18, '104': 19, '105': 20, '106': 21, '107': 22, '108': 23, '109': 24, '11': 25, '110': 26, '111': 27, '112': 28, '113': 29, '114': 30, '115': 31, '116': 32, '117': 33, '118': 34, '119': 35, '12': 36, '120': 37, '121': 38, '122': 39, '123': 40, '124': 41, '125': 42, '126': 43, '127': 44, '128': 45, '129': 46, '13': 47, '130': 48, '131': 49, '132': 50, '133': 51, '134': 52, '135': 53, '136': 54, '137': 55, '138': 56, '139': 57, '14': 58, '140': 59, '141': 60, '142': 61, '143': 62, '144': 63, '145': 64, '146': 65, '147': 66, '148': 67, '149': 68, '15': 69, '150': 70, '151': 71, '152': 72, '153': 73, '154': 74, '155': 75, '156': 76, '157': 77, '158': 78, '159': 79, '16': 80, '161': 81, '163': 82, '165': 83, '167': 84, '17': 85, '18': 86, '19': 87, '2': 88, '20': 89, '21': 90, '22': 91, '23': 92, '24': 93, '25': 94, '26': 95, '27': 96, '28': 97, '29': 98, '3': 99, '30': 100, '31': 101, '32': 102, '33': 103, '34': 104, '35': 105, '36': 106, '37': 107, '38': 108, '39': 109, '4': 110, '40': 111, '41': 112, '42': 113, '43': 114, '44': 115, '45': 116, '46': 117, '47': 118, '48': 119, '49': 120, '5': 121, '50': 122, '51': 123, '52': 124, '53': 125, '54': 126, '55': 127, '56': 128, '57': 129, '58': 130, '59': 131, '6': 132, '60': 133, '61': 134, '62': 135, '63': 136, '64': 137, '65': 138, '66': 139, '67': 140, '68': 141, '69': 142, '7': 143, '70': 144, '71': 145, '72': 146, '73': 147, '74': 148, '75': 149, '76': 150, '77': 151, '78': 152, '79': 153, '8': 154, '80': 155, '81': 156, '82': 157, '83': 158, '84': 159, '85': 160, '86': 161, '87': 162, '88': 163, '89': 164, '9': 165, '90': 166, '91': 167, '92': 168, '93': 169, '94': 170, '95': 171, '96': 172, '97': 173, '98': 174, '99': 175, 'B': 176, 'Br': 177, 'C': 178, 'Cl': 179, 'D': 180, 'F': 181, 'H': 182, 'I': 183, 'N': 184, 'O': 185, 'P': 186, 'S': 187, 'Si': 188, 'T': 189, '<sos>': 190, '<eos>': 191, '<pad>': 192}


if __name__ == "__main__":
    args = get_args()

    # file list dataframe
    logger.info("read dataframe.")
    if args.get("df") is None:
        df_train = pd.read_csv(f"{DIR_IN}train_labels.csv")
        df_files = pd.DataFrame(get_file_list(f"{DIR_IN}train/", regex_list=[r"\.png"]), columns=["filepath"])
        df_files["image_id"] = df_files["filepath"].str[-16:-4].copy()
        df_train = pd.merge(df_train, df_files, how="left", on="image_id")
        df_train.to_pickle("df_train.pickle")
    else:
        df_train = pd.read_pickle(args.get("df"))
    df_train["InChI"] = df_train["InChI"].str[9:]

    # validation
    ndf = np.random.permutation(df_train.index.values)
    df_valid = df_train.loc[ndf[:ndf.shape[0]//5 ]].copy()
    df_train = df_train.loc[ndf[ ndf.shape[0]//5:]].copy()

    # Dataset
    logger.info("create dataset.")
    def create_dataset(df, norm_mean=None, norm_std=None):
        return ImageDataset(
            df[["filepath", "InChI"]].apply(lambda x: x.to_dict(), axis=1).tolist(),
            str_filename="filepath", str_label="InChI",
            transforms=[
                ResizeFixRatio(IMAGE_IN_SIZE, fit_type="max"),
                transforms.CenterCrop(IMAGE_IN_SIZE),
                pil2cv,
                transforms.ToTensor(), #lambda x: x.astype(np.float32), これは定義しちゃだめ
                transforms.Normalize(mean=norm_mean[0], std=norm_std[0]),
            ]
        )
    if args.get("data") is None:
        norm_mean, norm_std = calc_normalize_mean_std(df_train["filepath"].values.tolist(), IMAGE_IN_SIZE, samples=10000, n_jobs=20)
        dataset_train = create_dataset(df_train, norm_mean[0], norm_std[0])
        dataset_valid = create_dataset(df_valid, norm_mean[0], norm_std[0])
        save_pickle(dataset_train, "./dataset_train.pickle")
        save_pickle(dataset_valid, "./dataset_valid.pickle")
    else:
        dataset_train = load_pickle("./dataset_train.pickle")
        dataset_valid = load_pickle("./dataset_valid.pickle")

    # DataLoader
    logger.info("create dataloader.")
    def collate_fn(batch, valid=False):
        _input, label = [], []
        for x, y in batch:
            _input.append(x.unsqueeze(0))
            label.append(y[0])
        _input = torch.cat(_input, dim=0)
        label  = tokenizer.texts_to_sequences(label)
        label  = torch.Tensor(label).to(torch.long)
        if valid:
            return (_input, ), label
        else:
            return (_input, label), label
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=64, shuffle=True, num_workers=0, 
        drop_last=True, collate_fn=partial(collate_fn, valid=False)
    )
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=32, shuffle=True, num_workers=0, 
        drop_last=True, collate_fn=partial(collate_fn, valid=True)
    )

    # Module
    class PositionalEncoding(torch.nn.Module):
        def __init__(self, d_model, dropout=0.1, max_len=5000):
            super(PositionalEncoding, self).__init__()
            self.dropout = torch.nn.Dropout(p=dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)
        def forward(self, x):
            x = x + self.pe[:x.size(0), :]
            return self.dropout(x)

    # Network
    class Net(torch.nn.Module):
        def __init__(self, embed_size: int, d_model: int, nhead: int, seq_length: int=512, index_ignore: int=None, index_sos: int=None, index_eos: int=None):
            super().__init__()
            self.seq_length   = seq_length
            self.index_ignore = index_ignore
            self.index_sos    = index_sos
            self.index_eos    = index_eos
            for x in [index_ignore, index_sos, index_eos]:
                if x is None: raise Exception("None")
            self.nn_encode    = timm.create_model("efficientnet_b2", pretrained=True)
            self.nn_encode.conv_stem = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.nn_encode_c  = torch.nn.Conv2d(1408, d_model, kernel_size=1, stride=1, padding=0, bias=False)
            self.nn_pos       = PositionalEncoding(d_model, dropout=0.1)
            self.nn_embed     = torch.nn.Embedding(embed_size, d_model)
            self.nn_mlp       = torch.nn.Linear(d_model, embed_size)
            self.nn_decode    = torch.nn.Transformer(
                d_model=d_model, nhead=nhead, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=2048, dropout=0.1, activation="relu",
            )
        def forward(self, _input, _label=None):
            output = _input.clone()
            output = self.nn_encode.forward_features(output)
            output = self.nn_encode_c(output)
            output = output.reshape(output.shape[0], output.shape[1], -1)
            output = torch.einsum("abc->cab", output) # (S, N, E) に変換
            if self.training:
                # training 
                label  = self.nn_embed(_label)
                label  = torch.einsum("abc->bac", label)  # (T, N, E) に変換
                tgt_key_padding_mask = (_label == self.index_ignore).to(_input.device)
                output = self.nn_decode(output, label, tgt_mask=self.nn_decode.generate_square_subsequent_mask(label.shape[0]).to(_input.device), tgt_key_padding_mask=tgt_key_padding_mask)
                output = torch.einsum("bac->abc", output)
                output = self.nn_mlp(output)
            else:
                # evaluation ( generate ). 共通する処理は多いが、まとめると分かりにくいため敢えて分ける
                _output = output.clone()
                _label  = torch.Tensor([[self.index_sos] for _ in range(_input.shape[0])]).to(torch.long).to(_input.device)
                for _ in range(self.seq_length):
                    # EOS 以降は PAD で埋める
                    indexes = torch.where(_label == self.index_eos)
                    for i, j in zip(indexes[0], indexes[1]):
                        _label[i, j+1:] = self.index_ignore
                    label  = self.nn_embed(_label)
                    label  = torch.einsum("abc->bac", label)  # (T, N, E) に変換
                    tgt_key_padding_mask = (_label == self.index_ignore).to(_input.device)
                    output = self.nn_decode(_output, label, tgt_mask=self.nn_decode.generate_square_subsequent_mask(label.shape[0]).to(_input.device), tgt_key_padding_mask=tgt_key_padding_mask)
                    output = torch.einsum("bac->abc", output)
                    output = self.nn_mlp(output)
                    # 予測したラベルを追加する
                    _label = torch.cat([_label, torch.argmax(output[:, -1, :], dim=1).reshape(-1, 1)], dim=1)
            return output
    
    # Loss
    class Loss(torch.nn.CrossEntropyLoss):
        def __init__(self, *args, index_ignore=None, index_sos=None, index_eos=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.index_ignore = index_ignore
            self.index_sos    = index_sos
            self.index_eos    = index_eos
            for x in [index_ignore, index_sos, index_eos]:
                if x is None: raise Exception("None")
        def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            print(torch.argmax(input, dim=2)[:10, :12], "\n", target[:10, 1:12+1])
            input       = input.reshape(-1, input.shape[-1])
            target      = target.reshape(-1)
            mask_ignore = (target != self.index_ignore)
            input       = input[ mask_ignore]
            target      = target[mask_ignore]
            input       = input [target != self.index_eos]
            target      = target[target != self.index_sos]
            return super().forward(input, target)
    
    # Train
    net = Net(len(tokenizer), 256, 16, seq_length=SEQ_LENGTH, index_ignore=tokenizer.stoi["<pad>"], index_sos=tokenizer.stoi["<sos>"], index_eos=tokenizer.stoi["<eos>"])
    trainer = BaseNN(
        net, mtype="cls",
        loss_funcs=Loss(index_ignore=tokenizer.stoi["<pad>"], index_sos=tokenizer.stoi["<sos>"], index_eos=tokenizer.stoi["<eos>"]),
        optimizer=torch.optim.SGD, optim_params={"lr":0.01, "weight_decay":0},
        dataloader_train=dataloader_train,
        #dataloader_valids=[dataloader_valid], valid_step=10,
        print_step=200
    )
    if args.get("load") is not None:
        trainer.load(model_path=args.get("load"))
    trainer.to_cuda()
    trainer.train()
