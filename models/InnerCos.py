import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import util.util as util

# import networks from the pix2pix project used for the encoder located in ./l1encoder/
import l1encoder.models.networks as networks

def verify(values):
    # just to make sure we have values between -1 and 1
    minn = torch.amin(values)
    maxx = torch.amax(values)
    if minn < -1.01 or maxx > 1.01:
        raise Exception("Values were not bound by a tanh where this was expected")
    return values

def load_network(checkpoint_dir, n_downsampling, device, gpu_ids):
    net = networks.define_G(input_nc=3, output_nc=3, ngf=32, netG=None, norm='batch', use_dropout=False,
                                init_type='normal', init_gain=0.02, gpu_ids=gpu_ids, n_downsampling=n_downsampling)
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    print('loading the model from %s' % checkpoint_dir)
    state_dict = torch.load(checkpoint_dir + '/latest_net_G.pth', map_location=str(device))
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    net.load_state_dict(state_dict)
    print("Loaded frozen network weights into model.")
    for param in net.parameters():
        param.requires_grad = False
    net.eval() # to switch batch norm to testing mode (no running means and variance)
    return net

class InnerCos(nn.Module):
    def __init__(self, original=True, device=None, gpu_ids=[]):
        super(InnerCos, self).__init__()
        self.criterion = nn.L1Loss()
        self.target = None
        self.original = original

        print("ORIGINAL MODE", original)
        if original:
            self.down_model = nn.Sequential(
                nn.Conv2d(256, 3, kernel_size=1,stride=1, padding=0),
                nn.Tanh()
            )
        else:
            # use our variant, load the network and the pre-trained weights
            checkpoint_dir = "/home/scriptandhands/MEDFE/Rethinking-Inpainting-MEDFE/l1encoder/checkpoints/"
            self.down_model_structures = load_network(checkpoint_dir + "celeba_8", n_downsampling=5, device=device, gpu_ids=gpu_ids)
            self.down_model_textures = load_network(checkpoint_dir + "celeba_32", n_downsampling=3, device=device, gpu_ids=gpu_ids)

    def set_target(self, targetde, targetst):
        # --st_root=[the path of structure images]
        # --de_root=[the path of ground truth images]

        # original operates on [3, 32, 32], our variant operates on [256, 32, 32]
        if self.original:
            self.targetst = F.interpolate(targetst, size=(32, 32), mode='bilinear')
            self.targetde = F.interpolate(targetde, size=(32, 32), mode='bilinear')
        else:
            # the self.down_model_* models return [encoded, decoded]
            with torch.no_grad():
                # structure images are no longer loaded seperately
                self.targetst = verify(self.down_model_structures(targetde)[0].detach())
                self.targetde = verify(self.down_model_textures(targetde)[0].detach())
            

    def get_target(self):
        return self.target

    def forward(self, in_data):
        loss_co = in_data[1]

        if self.original:
            self.ST = self.down_model(loss_co[0])
            self.DE = self.down_model(loss_co[1])
        else:
            self.ST = verify(loss_co[0])
            self.DE = verify(loss_co[1])

        # print('comparing sizes', self.ST.shape, 'and', self.targetst.shape)
        self.loss = self.criterion(self.ST, self.targetst) + self.criterion(self.DE, self.targetde)
        self.output = in_data[0]
        return self.output

    def get_loss(self):
        return self.loss

    def __repr__(self):
        return self.__class__.__name__