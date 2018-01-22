import argparse
import os
import torch
from torch.utils import data
import time
import logging
from tqdm import tqdm

from loader_hdf5 import LoaderHdf5
from loader_numpy import LoaderNumpy
from loader_png import LoaderPng

if __name__ == '__main__':
    #
    #   load configurations from
    #

    parser = argparse.ArgumentParser(description='Train DeepTrack')
    parser.add_argument('-d', '--dataset', help="Dataset path", metavar="FILE")
    parser.add_argument('-o', '--log', help="log path, (will write in a file with HourMinuteSeconds.log)",
                        metavar="FILE", default="./")
    parser.add_argument('-i', '--device', help="Gpu id", action="store", default=0, type=int)
    parser.add_argument('-k', '--backend', help="backend : cuda | cpu", action="store", default="cuda")
    parser.add_argument('-s', '--batchsize', help="Size of minibatch", action="store", default=64, type=int)
    parser.add_argument('-e', '--epoch', help="number of epoch", action="store", default=1, type=int)
    parser.add_argument('-m', '--sharememory', help="Activate share memory", action="store_true")
    parser.add_argument('-n', '--ncore', help="number of cpu core to use, -1 is all core", action="store", default=-1,
                        type=int)

    arguments = parser.parse_args()

    device_id = arguments.device
    backend = arguments.backend
    batch_size = arguments.batchsize
    epoch = arguments.epoch
    use_shared_memory = arguments.sharememory
    number_of_core = arguments.ncore

    # Setup paths and logging
    data_path = os.path.expandvars(arguments.dataset)
    log_path = os.path.expandvars(arguments.log)
    logging.basicConfig(
        format='%(levelname)s:%(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(log_path, time.strftime("%Hh%Mm%Ss.log"))),
            logging.StreamHandler()
        ])

    # Cheap way to retrieve the loader type by verifying file extension in folder
    extension = os.listdir(data_path)[0].split(".")[-1]
    logging.info("Loading {} dataset...".format(extension))
    if extension == "png":
        loader_class = LoaderPng
    elif extension == "npy":
        loader_class = LoaderNumpy
    elif extension == "hdf5":
        loader_class = LoaderHdf5
    else:
        raise RuntimeError("Dataset Extension {} is not supported".format(extension))

    # setup backend and number of cpu core to use for loading
    if backend == "cuda":
        torch.cuda.set_device(device_id)
    if number_of_core == -1:
        number_of_core = os.cpu_count()

    # Setup dataset objects (will index it)
    index_start_time = time.time()
    dataset = loader_class(data_path)
    logging.info("Indexing time : {}".format(time.time() - index_start_time))
    train_loader = data.DataLoader(dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=number_of_core,
                                   pin_memory=use_shared_memory,
                                   drop_last=True,
                                   )

    # Run e epoch, load all data and transfer the minibatch on gpu memory
    total_start_time = time.time()
    for e in range(epoch):
        epoch_start_time = time.time()
        for i, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            rgb_tensor, depth_tensor = data[0]
            if backend == "cuda":
                rgb_tensor = rgb_tensor.cuda()
                depth_tensor = depth_tensor.cuda()

        logging.info("Epoch load time : {}".format(time.time() - epoch_start_time))
    logging.info("Total load time : {}".format(time.time() - total_start_time))

