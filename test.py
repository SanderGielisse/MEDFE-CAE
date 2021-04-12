import time
import pdb
from options.test_options import TestOptions
from data.dataprocess import DataProcess
from models.models import create_model
import torchvision
from torch.utils import data
#from torch.utils.tensorboard import SummaryWriter
import os
import torch
from PIL import Image
import numpy as np
from glob import glob
from tqdm import tqdm
import torchvision.transforms as transforms
if __name__ == "__main__":

    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    opt = TestOptions().parse()
    model = create_model(opt)

    # or pth with 
    model.netEN.module.load_state_dict(torch.load("./checkpoints/" + opt.name + "/EN.pth")['net'])
    model.netDE.module.load_state_dict(torch.load("./checkpoints/" + opt.name + "/DE.pth")['net'])
    model.netMEDFE.module.load_state_dict(torch.load("./checkpoints/" + opt.name + "/MEDFE.pth")['net'])
    # results_dir = r'./results'
    # if not os.path.exists( results_dir):
    #     os.mkdir(results_dir)

    # mask_paths = sorted(glob('{:s}/*'.format(opt.mask_root)))
    de_paths = sorted(glob('{:s}/*'.format(opt.de_root)))
    st_paths = sorted(glob('{:s}/*'.format(opt.st_root)))

    mask = torch.empty([3, 256, 256], dtype=torch.float32) # self.mask_transform(mask_img.convert('RGB'))
    mask[:, :, :] = 0.0
    mask[:, 64:(128+64), 64:(128+64)] = 1.0
    mask = torch.unsqueeze(mask, 0)

    image_len = 2048
    for i in tqdm(range(image_len)):
        # only use one mask for all image
        # path_m = mask_paths[0]
        path_d = de_paths[i]
        path_s = de_paths[i]

        # mask = Image.open(path_m).convert("RGB")
        detail = Image.open(path_d).convert("RGB")
        structure = Image.open(path_s).convert("RGB")

        # mask = mask_transform(mask)
        detail = img_transform(detail)
        structure = img_transform(structure)
        detail = torch.unsqueeze(detail, 0)
        structure = torch.unsqueeze(structure, 0)

        with torch.no_grad():
            model.set_input(detail, structure, mask)
            model.forward()
            fake_out = model.fake_out
            fake_out = fake_out.detach().cpu() * mask + detail*(1-mask)
            fake_image = (fake_out + 1) / 2.0

        output = fake_image.detach().numpy()[0].transpose((1, 2, 0))*255
        output = Image.fromarray(output.astype(np.uint8))
        output.save(rf"{opt.results_dir}/{opt.name}/{i}.png")

        o = (detail + 1) / 2.0
        o = o.detach().numpy()[0].transpose((1, 2, 0))*255
        o = Image.fromarray(o.astype(np.uint8))
        o.save(rf"{opt.results_dir}/ground_truth/{i}.png")
        # print("Done %d" % i)