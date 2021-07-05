import os
import numpy as np
import cv2
from functools import partial
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.transforms as trf

# local package
from kkutils.lib.nn.model import BaseNN
from kkutils.lib.nn.module import TorchNN, Layer
from kkutils.lib.nn.model_image import ImageDataset
from kkutils.util.image.utils import pil2cv, bgr_to_gray
from kkutils.util.image.transform import ResizeFixRatio
from kkutils.util.com import get_file_list


INSIZE=32*8


def work(ndf: np.ndarray, bins=[0, 76, 86, 95, 105, 255]):
    return (np.digitize(ndf, bins=bins) - 1) / (len(bins) - 2)

def get_lookup_table(length, n_dimension):
    lt = np.array([
        [pos / np.power(10000, 2.0 * i / n_dimension) for i in range(int(n_dimension / 2))]
        for pos in range(length)
    ])
    lt = np.repeat(lt, 2, axis = 1)
    lt[:, 0::2] = np.sin(lt[:, 0::2])
    lt[:, 1::2] = np.cos(lt[:, 1::2])
    return lt

class DividePatch(nn.Module):
    def __init__(self, n_pixel_h: int, n_pixel_w: int):
        super().__init__()
        self.n_pixel_h = n_pixel_h
        self.n_pixel_w = n_pixel_w
    def forward(self, input: torch.Tensor):
        # input (B, C, H, W). Batch, Channel, Height, Width
        output = torch.split(input, self.n_pixel_h, dim=2)
        output = [torch.split(tens, self.n_pixel_w, dim=3) for tens in output]
        output = [y for x in output for y in x]
        output = torch.stack(output, dim=0)
        output = torch.einsum("pbchw->bpchw", output)
        output = output.reshape(output.shape[0], output.shape[1], -1)
        return output

class MLP(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.mlp1 = nn.Linear(n_in, n_out)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(n_in, n_out)
    def forward(self, input: torch.Tensor):
        output = self.mlp1(input)
        output = self.gelu(output)
        output = self.mlp2(output)
        return output

class MixerLayer(nn.Module):
    def __init__(self, n_channel: int, n_patch: int):
        super().__init__()
        self.norm = nn.LayerNorm(n_channel)
        self.mlp1 = MLP(n_patch, n_patch)
        self.mlp2 = MLP(n_channel, n_channel)
    def forward(self, input: torch.Tensor):
        input   = input.clone()
        output  = self.norm(input)
        output  = torch.einsum("abc->acb", output) # Transpose
        output  = self.mlp1(output)
        output  = torch.einsum("abc->acb", output) # Transpose
        output  = output + input
        _output = output.clone()
        output  = self.norm(output)
        output  = self.mlp2(output)
        output  = output + _output
        return output

class MLPMixer(nn.Module):
    def __init__(self, image_size_h: int, image_size_w: int, n_patch_h: int, n_patch_w: int, n_out_fc: int=None, n_org_channel: int=3, n_layers: int=1):
        super().__init__()
        assert image_size_h % n_patch_h == 0
        assert image_size_w % n_patch_w == 0
        self.n_org_channel = n_org_channel
        self.n_patch_h     = n_patch_h
        self.n_patch_w     = n_patch_w
        self.divide_patch  = DividePatch(int(image_size_h // n_patch_h), int(image_size_w // n_patch_w))
        n_in               = int(self.divide_patch.n_pixel_h * self.divide_patch.n_pixel_w * n_org_channel)
        n_out              = n_in if n_out_fc is None else n_out_fc
        self.pre_patch_fc  = nn.Linear(n_in, n_out)
        self.layers        = nn.ModuleList([MixerLayer(n_out, n_patch_h * n_patch_w) for _ in range(n_layers)])
    def forward(self, input: torch.Tensor):
        output = self.divide_patch(input)
        output = self.pre_patch_fc(output)
        for _layer in self.layers:
            output = _layer(output)
        output = output.reshape(output.shape[0], output.shape[1], self.n_org_channel, self.divide_patch.n_pixel_h, self.divide_patch.n_pixel_w)
        output = torch.einsum("bpchw->pbchw", output)
        output = [torch.cat([output[self.n_patch_w*j+i] for i in range(self.n_patch_w)], dim=3) for j in range(self.n_patch_h)]
        output = torch.cat(output, dim=2)
        return output


if __name__ == "__main__":
    dirpath = "../input/carpet/train/good/"
    flist   = get_file_list(dirpath, regex_list=[r"\.png"])
    #flist   = flist[0:128]
    dataset_train = ImageDataset(
        [{"name": os.path.basename(x), "label": 0} for x in flist],
        root_dirpath=dirpath,
        transforms=[
            ResizeFixRatio(INSIZE),
            trf.CenterCrop((INSIZE, INSIZE)),
            pil2cv,
            bgr_to_gray,
        ]
    )
    dirpath = "../input/carpet/test/good/"
    flist   = get_file_list(dirpath, regex_list=[r"\.png"])
    dataset_test = ImageDataset(
        [{"name": os.path.basename(x), "label": 0} for x in flist],
        root_dirpath=dirpath,
        transforms=[
            ResizeFixRatio(INSIZE),
            trf.CenterCrop((INSIZE, INSIZE)),
            pil2cv,
            bgr_to_gray,
        ]
    )
    
    #ndf = np.concatenate([x for x, _ in dataset_train]).reshape(-1)
    #__work = partial(work, bins=[0, np.percentile(ndf, 50), 255])
    #plt.hist(ndf, bins=50)
    #plt.savefig("hist.png")
    def collate_fn(batch):
        images, _= list(zip(*batch))
        #images   = torch.stack([torch.from_numpy(x.reshape(-1).astype(float) / 255.).to(torch.float) for x in images])
        #images   = torch.stack([torch.from_numpy(__work(x.reshape(-1).astype(float))).to(torch.float) for x in images])
        images   = torch.stack([torch.from_numpy(x.astype(float) / 255.).to(torch.float).unsqueeze(0) for x in images])
        return images, images.clone()
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=16, shuffle=True, num_workers=0, 
        drop_last=True, collate_fn=collate_fn
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0, 
        drop_last=True, collate_fn=collate_fn
    )

    class MyMod(nn.Module):
        def __init__(self, width, height, channel):
            self.width   = width
            self.height  = height
            self.channel = channel
            super().__init__()
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input.reshape(-1, self.channel, self.height, self.width)
    class MyMod2(nn.Module):
        def __init__(self, size):
            x = get_lookup_table(size, 2)[:, 0]
            self.x = torch.from_numpy(x).to(torch.float).to("cuda")
            super().__init__()
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input + self.x
    """
    mynn = TorchNN(
        INSIZE**2,
        #Layer("", MyMod2(INSIZE**2), None, None, (), {}             ),
        Layer("", nn.Linear, INSIZE**2*8,  None, (), {"bias": False}),
        Layer("", nn.ReLU,           None, None, (), {}             ),
        Layer("", nn.Linear, INSIZE**2,    None, (), {"bias": False}),
        #Layer("", nn.Sigmoid,        None, None, (), {}             ),
    )
    """
    """
    mynn = TorchNN(
        INSIZE**2,
        Layer("", MyMod2(INSIZE**2), None, None, (), {}          ),
        Layer("", nn.Linear, INSIZE**2, None, (), {"bias": False}),
        Layer("", nn.ReLU,        None, None, (), {}             ),
        Layer("", nn.Linear, INSIZE**2, None, (), {"bias": False}),
        Layer("", nn.ReLU,        None, None, (), {}             ),
        Layer("", nn.Linear, INSIZE**2, None, (), {"bias": False}),
        Layer("", nn.ReLU,        None, None, (), {}             ),
        Layer("", nn.Linear, INSIZE**2, None, (), {"bias": False}),
        Layer("", nn.ReLU,        None, None, (), {}             ),
        Layer("", nn.Linear, INSIZE**2, None, (), {"bias": False}),
        #Layer("", nn.Sigmoid,        None, None, (), {}             ),
    )
    """
    """
    mynn = TorchNN(
        INSIZE**2,
        Layer("", nn.Linear,   INSIZE**2*8, None, (), {"bias": False}),
        Layer("", nn.ReLU,            None, None, (), {}             ),
        Layer("", MyMod,              None, None, (24, 24, 128), {}),
        Layer("", nn.ConvTranspose2d(128, 1, kernel_size=4, stride=4), None, None, (), {}),
        Layer("", nn.Identity,        None, "reshape(x,-1)", (), {} ),
    )
    """
    """
    mynn = TorchNN(
        0,
        Layer("", MyMod,              None, None, (INSIZE, INSIZE, 1), {}),
        Layer("", nn.Conv2d(1, 128, kernel_size=4, stride=4), None, None, (), {}),
        Layer("", nn.ReLU,            None, None, (), {} ),
        Layer("", nn.ConvTranspose2d(128, 1, kernel_size=4, stride=4), None, None, (), {}),
        Layer("", nn.Identity,        None, "reshape(x,-1)", (), {} ),
    )
    """
    """
    mynn = TorchNN(
        0,
        Layer("", MyMod,              None, None, (INSIZE, INSIZE, 1), {}),
        Layer("", nn.Conv2d(  1, 128, kernel_size=4, stride=4), None, None, (), {}),
        Layer("", nn.ReLU,            None, None, (), {} ),
        Layer("", nn.Conv2d(128, 512, kernel_size=4, stride=4), None, None, (), {}),
        Layer("", nn.ReLU,            None, None, (), {} ),
        Layer("", nn.Conv2d(512,1024*18,kernel_size=6, stride=1), None, None, (), {}),
        Layer("", nn.ReLU,            None, None, (), {} ),
        Layer("", nn.ConvTranspose2d(1024*18,512, kernel_size=6, stride=1), None, None, (), {}),
        Layer("", nn.ReLU,            None, None, (), {} ),
        Layer("", nn.ConvTranspose2d(512, 128, kernel_size=4, stride=4), None, None, (), {}),
        Layer("", nn.ReLU,            None, None, (), {} ),
        Layer("", nn.ConvTranspose2d(128,   1, kernel_size=4, stride=4), None, None, (), {}),
        Layer("", nn.Identity,        None, "reshape(x,-1)", (), {} ),
    )
    """
    """
    mynn = TorchNN(
        0,
        Layer("", MyMod,              None, None, (INSIZE, INSIZE, 1), {}),
        Layer("", nn.Conv2d(  1, 128, kernel_size=4, stride=4), None, None, (), {}),
        Layer("", nn.ReLU,            None, None, (), {} ),
        Layer("", nn.Conv2d(128, 256, kernel_size=4, stride=4), None, None, (), {}),
        Layer("", nn.ReLU,            None, None, (), {} ),
        Layer("", nn.Identity,        None, "reshape(x,-1)", (), {} ),
        Layer("", nn.Linear(9216, 9216, bias=False), None, None, (), {}),
        Layer("", MyMod,              None, None, (6, 6, 256), {}),
        Layer("", nn.ConvTranspose2d(256, 128, kernel_size=4, stride=4), None, None, (), {}),
        Layer("", nn.ReLU,            None, None, (), {} ),
        Layer("", nn.ConvTranspose2d(128,   1, kernel_size=4, stride=4), None, None, (), {}),
        Layer("", nn.Identity,        None, "reshape(x,-1)", (), {} ),
    )
    """
    mynn = MLPMixer(INSIZE, INSIZE, int(INSIZE // 16), int(INSIZE // 16), n_org_channel=1, n_layers=3)
    trainer = BaseNN(
        mynn,
        #loss_funcs=torch.nn.SmoothL1Loss(beta=50/255),
        #loss_funcs=torch.nn.BCELoss(),
        loss_funcs=torch.nn.MSELoss(),
        optimizer = torch.optim.SGD, 
        optim_params = { "lr": 0.5,"weight_decay": 0 }, 
        dataloader_train=dataloader_train,
        dataloader_valids=[dataloader_test, ],
        valid_step=10, epoch=10000
    )
    #trainer.load("./output_bce/model_3594.pth")
    trainer.to_cuda()
    trainer.train()

    output, _ = trainer.predict_proba_dataloader(dataloader=dataloader_train)
    output    = (output[0][0] * 255).astype(np.uint8)
    cv2.imwrite("test0.png", output.reshape(INSIZE, INSIZE))
    cv2.imwrite("test1.png", dataset_train[0][0])
    #cv2.imwrite("test2.png", (__work(dataset_train[0][0]) * 255).astype(np.uint8))
    output, _ = trainer.predict_proba_dataloader(dataloader=dataloader_test)
    output    = (output[0][0] * 255).astype(np.uint8)
    cv2.imwrite("test3.png", output.reshape(INSIZE, INSIZE))
    cv2.imwrite("test4.png", dataset_test[0][0])
    #cv2.imwrite("test5.png", (__work(dataset_test[0][0]) * 255).astype(np.uint8))
