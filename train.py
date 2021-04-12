import time
from options.train_options import TrainOptions
from data.dataprocess import DataProcess
from models.models import create_model
import torchvision
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import os
import torch
if __name__ == "__main__":

    opt = TrainOptions().parse()
    # define the dataset
    print("Define dataset...")
    dataset = DataProcess(opt.de_root,opt.st_root,opt.mask_root,opt,opt.isTrain)
    print("Building iterator...")
    iterator_train = (data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.num_workers))
    # Create model
    print("Creating model...")
    model = create_model(opt)
    print("Model created.")
    total_steps=0
    # Create the logs
    dir = os.path.join(opt.log_dir, opt.name).replace('\\', '/')
    if not os.path.exists(dir):
        os.mkdir(dir)
    print("Creating writer...")
    writer = SummaryWriter(log_dir=dir, comment=opt.name)
    print("Writer created.")
    # Start Training
    for epoch in range (opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        # print("Loading batch...")
        for detail, structure, mask in iterator_train:
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            # print("Setting input...")
            model.set_input(detail, structure, mask)
            model.optimize_parameters()
            # display the training processing
            if total_steps % opt.display_freq == 0:
                input, output, GT = model.get_current_visuals()
                image_out = torch.cat([input, output, GT], 0)
                grid = torchvision.utils.make_grid(image_out)
                writer.add_image('Epoch_(%d)_(%d)' % (epoch, total_steps + 1), grid, total_steps + 1)
            # display the training loss
            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                dict = {}
                for sc in ['G_GAN', 'G_L1', 'G_stde', 'D', 'F']:
                    val = float(errors[sc])
                    writer.add_scalar(sc, val, total_steps + 1)
                    dict[sc] = val
                print('iteration time (seconds): %.3f' % t, dict)
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks(epoch)
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
    writer.close()
