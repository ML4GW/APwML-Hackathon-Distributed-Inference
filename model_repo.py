import argparse
import sys
import os
import torch

cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# our hermes imports
from hermes import quiver as qv
from hermes.aeriel.client import InferenceClient
from hermes.aeriel.serve import serve
from hermes.stillwater import ServerMonitor

class GlobalAvgPool(torch.nn.Module):
    def forward(self, x):
        return x.mean(axis=-1)
        
def main(**kwargs):

    repo_path=kwargs["path"]

    NUM_IFOS = 2  # number of interferometers analyzed by our model
    SAMPLE_RATE = 2048  # rate at which input data to the model is sampled
    KERNEL_LENGTH = 4  # length of the input to the model in seconds

    # inference parameters
    INFERENCE_DATA_LENGTH = 2048  # amount of data to analyze at inference time
    INFERENCE_SAMPLING_RATE = 0.25  # rate at which we'll sample input windows from the inference data
    INFERENCE_RATE = 250  # seconds of data we'll try to analyze per second

    # convert some of these into more useful units for slicing purposes
    kernel_size = int(SAMPLE_RATE * KERNEL_LENGTH)
    inference_stride = int(SAMPLE_RATE / INFERENCE_SAMPLING_RATE)
    inference_data_size = int(SAMPLE_RATE * INFERENCE_DATA_LENGTH)
    num_inferences = (inference_data_size - kernel_size) // inference_stride + 1

    # limit the number of requests we make per second
    # so that we don't overload the network or server
    kernels_per_second = int(INFERENCE_RATE * INFERENCE_SAMPLING_RATE)
    rate_limiter = RateLimiter(max_calls=kernels_per_second, period=1)

    class GlobalAvgPool(torch.nn.Module):
        def forward(self, x):
            return x.mean(axis=-1)


    nn = torch.nn.Sequential(
        torch.nn.Conv1d(NUM_IFOS, 8, kernel_size=7, stride=2),
        torch.nn.ReLU(),
        torch.nn.Conv1d(8, 32, kernel_size=7, stride=2),
        torch.nn.ReLU(),
        torch.nn.Conv1d(32, 64, kernel_size=7, stride=2),
        torch.nn.ReLU(),
        torch.nn.Conv1d(64, 128, kernel_size=7, stride=2),
        torch.nn.ReLU(),
        torch.nn.Conv1d(128, 256, kernel_size=7, stride=2),
        torch.nn.ReLU(),
        GlobalAvgPool(),
        torch.nn.Linear(256, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 1)
        )

    # let's make sure we're starting with a fresh repo
    utils.clear_repo(repo_path)

    # initialize a blank model repository
    repo = qv.ModelRepository(repo_path)
    assert len(repo.models) == 0  # this attribute will get updated as we add models

    # create a new entry in the repo for our model
    model = repo.add("my-classifier", platform=qv.Platform.ONNX)
    assert len(repo.models) == 1
    assert model == repo.models["my-classifier"]

    # now export our current version of the network to this entry.
    # Since we haven't exported any versions of this model yet,
    # Triton needs to know what names to give the inputs and
    # outputs and what shapes to expect, so we have to specify
    # them explicitly this first time.
    # Note that -1 indicates variable length batch dimension.
    model.export_version(
        nn,
        input_shapes={"hoft": (-1, NUM_IFOS, kernel_size)},
        output_names=["prob"]
    )



# python initialisation -p /home/vasileios.skliris/testInit
if __name__ == "__main__":
    
    arguments = ["path"]
    
    #Construct argument parser:
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str
                        , default = "./my_model"
                        , help = "Path for the model"
                        , required = False)

    
    # Pass arguments:
    args = parser.parse_args()
    
    # Store arguments in dictionary:
    kwargs = {}
    for argument in arguments:
        kwargs[argument] = getattr(args, argument)

    main(**kwargs)