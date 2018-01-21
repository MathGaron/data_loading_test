import argparse
import os
import torch
from torch.utils import data
import time

from loader_png import LoaderPng

if __name__ == '__main__':
    #
    #   load configurations from
    #

    parser = argparse.ArgumentParser(description='Train DeepTrack')
    parser.add_argument('-o', '--output', help="Output path", metavar="FILE")
    parser.add_argument('-d', '--dataset', help="Dataset path", metavar="FILE")
    parser.add_argument('-i', '--device', help="Gpu id", action="store", default=0, type=int)
    parser.add_argument('-k', '--backend', help="backend : cuda | cpu", action="store", default="cuda")
    parser.add_argument('-s', '--batchsize', help="Size of minibatch", action="store", default=64, type=int)
    parser.add_argument('-m', '--sharememory', help="Activate share memory", action="store_true")
    parser.add_argument('-n', '--ncore', help="number of cpu core to use, -1 is all core", action="store", default=-1,
                        type=int)

    arguments = parser.parse_args()

    device_id = arguments.device
    backend = arguments.backend
    batch_size = arguments.batchsize
    use_shared_memory = arguments.sharememory
    number_of_core = arguments.ncore

    data_path = os.path.expandvars(arguments.dataset)
    output_path = os.path.expandvars(arguments.output)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if backend == "cuda":
        torch.cuda.set_device(device_id)
    if number_of_core == -1:
        number_of_core = os.cpu_count()

    index_start_time = time.time()
    dataset = LoaderPng(data_path)
    print("Indexing time : {}".format(time.time() - index_start_time))

    train_loader = data.DataLoader(dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=number_of_core,
                                   pin_memory=use_shared_memory,
                                   drop_last=True,
                                   )

    epoch_start_time = time.time()
    for i, (data, target) in enumerate(train_loader):
        rgb_tensor, depth_tensor = data[0]

        # process...
    print("Epoch load time : {}".format(time.time() - epoch_start_time))
