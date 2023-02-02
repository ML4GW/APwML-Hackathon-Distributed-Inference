import argparse
import sys
import os

from queue import Queue

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





def main(output_queue, ip ,data = None,model_name = "ensemble", batch_size=32, ):

    ip = ip.split(":")[0]+":8001"
    openingclientT0 = time.time()
    client = InferenceClient(
            ip,
            model_name="ensemble",
            model_version=1,
            batch_size=batch_size)

    print("OPENING CLIENT TIME: ",time.time()-openingclientT0)

    with client:#, monitor:

        t0datasetget = time.time()            

        # model parameters
        NUM_IFOS = 3  # number of interferometers analyzed by our model
        SAMPLE_RATE = 1  # rate at which input data to the model is sampled
        KERNEL_LENGTH = 1  # length of the input to the model in seconds

        # inference parameters
        INFERENCE_DATA_LENGTH = batch_size*128 # 8192 * 16  # amount of data to analyze at inference time
        INFERENCE_SAMPLING_RATE = 1  # rate at which we'll sample input windows from the inference data
        INFERENCE_RATE = 12000  # seconds of data we'll try to analyze per second

        kernel_size = int(SAMPLE_RATE * KERNEL_LENGTH)
        inference_stride = int(SAMPLE_RATE / INFERENCE_SAMPLING_RATE)
        inference_data_size = int(SAMPLE_RATE * INFERENCE_DATA_LENGTH)

        # define some parameters which apply to both data streams
        num_kernels = inference_data_size // inference_stride
        num_inferences = num_kernels // batch_size
        kernels_per_second = INFERENCE_RATE * INFERENCE_SAMPLING_RATE
        batches_per_second = kernels_per_second / batch_size

        time_exportS0=time.time()

        strainList      = np.random.randn(batch_size*128,1024,NUM_IFOS).astype("float32")
        correlationList = np.random.randn(batch_size*128, 60, NUM_IFOS).astype("float32")

        print(strainList.shape,correlationList.shape)
        print(strainList[0].shape)

        inferencet0 = time.time()

        for i in range(int(num_inferences)):
            #print(i,'/',num_inferences)
            start = i * 1 * batch_size
            stop = start + 1 * batch_size
            c_start = i * 1 * batch_size
            c_stop = c_start + 1 * batch_size

            kernel = {'strain' : strainList[ start : stop ]
                        ,'correlation' : correlationList[ c_start : c_stop ]}

            print(strainList[ start : stop ].shape)
            print(correlationList[ c_start : c_stop ].shape)
            

            client.infer(kernel, request_id=i)
            print('inference requested')
            # sleep to roughly maintain our inference rate
            time.sleep(0.9 / batches_per_second)

            if i < 5:
                response_ = client.get()
                trials = 1
                while response_ is None and trials<5000:
                    response_ = client.get()
                    time.sleep(1e-2)
                    trials+=1
                    if trials>=5000: raise(Exception("Trials exceeded 5000."))

                response, _, __ = response_
                #print(i,'response type ', type(response) , response.shape , trials)
                output_queue.put(response)
                print(i, " inference complete")
        for i in range(int(num_inferences)-5):
            trials = 1
            response_ = client.get()
            while response_ is None:
                time.sleep(1e-2)
                trials+=1
                response_ = client.get()
                if trials>=5000: raise(Exception("Trials exceeded 5000."))

            response, _, __ = response_

            print(response)

            output_queue.put(response)
            print(i, " inference complete")





if __name__ == "__main__":


    output = Queue()

    arguments = ["ip"]

    #Construct argument parser:
    parser = argparse.ArgumentParser()

    parser.add_argument('-ip', '--ip', type=str
                        , default = "localhost:8001"
                        , help = "IP of already running server"
                        , required = True)

    # Pass arguments:
    args = parser.parse_args()

    # Store arguments in dictionary:
    kwargs = {}
    for argument in arguments:
        kwargs[argument] = getattr(args, argument)

    main(output, **kwargs)
