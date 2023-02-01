import argparse
import sys
import os

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



def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="path to model",
    )

    parser.add_argument(
        "-d",
        "--data",
        required=True,
        help="path to data",
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        required=True,
        help="batch size",
    )

    parser.add_argument(
        "-s",
        "--submit_dir",
        default=None,
        help="submit directory",
    )

    parser.add_argument(
        "-l",
        "--log_dir",
        default=None,
        help="log directory",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        default=None,
        help="output directory",
    )

    parser.add_argument(
        "-e",
        "--error_dir",
        default=None,
        help="error directory",
    )

    return parser

parser = create_arg_parser()

args = parser.parse_args(sys.argv[1:])

print(args)
print(args.model)

# let's deny TensorFlow any GPUs up front so that it doesn't
# take up all of our available GPU memory. We don't need GPUs
# for export anyway, so it shouldn't affect anything




def cleanup_df(df, t0):
    df["Time since start (s)"] = df["timestamp"] - t0
    df["Average queue time (us)"] = df["queue"] / df["count"]

    # count all inference steps as a single metric
    # of inference latency
    infer_time = df[[f"compute_{i}" for i in ["input", "infer", "output"]]].sum(axis=1)
    df["Average infer time (us)"] = infer_time / df["count"]

    # use the number of inferences completed in an interval
    # along with the inference sampling rate to put throughput
    # in units of data seconds per second
    df["Throughput (s' / s)"] = batch_size * df["count"] / df["timestamp"].diff() / INFERENCE_SAMPLING_RATE

    return df[[
        "Time since start (s)",
        "Throughput (s' / s)",
        "Average queue time (us)",
        "Average infer time (us)"
    ]]


df = pd.read_csv(metrics_file)
dfs = []
for model, subdf in df.groupby("model"):
    subdf = cleanup_df(subdf, df.timestamp.min()).iloc[1:]
    subdf["model"] = model
    dfs.append(subdf)

df = pd.concat(dfs, ignore_index=True)
df.to_csv(metrics_dir / "non-streaming_single-model_clean.csv", index=False)





def main(**args):

    # model parameters
    NUM_IFOS = 3  # number of interferometers analyzed by our model
    SAMPLE_RATE = 1024  # rate at which input data to the model is sampled
    KERNEL_LENGTH = 1  # length of the input to the model in seconds

    # inference parameters
    INFERENCE_DATA_LENGTH = 8192 * 16  # amount of data to analyze at inference time
    INFERENCE_SAMPLING_RATE = 1  # rate at which we'll sample input windows from the inference data
    INFERENCE_RATE = 12000  # seconds of data we'll try to analyze per second

    # CORRELATION inference parameters
    C_SAMPLE_RATE = 1  # rate at which input data to the model is sampled
    C_KERNEL_LENGTH = 1  # length of the input to the model in seconds

    # convert some of these into more useful units for slicing purposes
    batch_size = 16

    kernel_size = int(SAMPLE_RATE * KERNEL_LENGTH)
    inference_stride = int(SAMPLE_RATE / INFERENCE_SAMPLING_RATE)
    inference_data_size = int(SAMPLE_RATE * INFERENCE_DATA_LENGTH)

    # define some parameters which apply to both data streams
    num_kernels = inference_data_size // inference_stride
    num_inferences = num_kernels // batch_size
    kernels_per_second = INFERENCE_RATE * INFERENCE_SAMPLING_RATE
    batches_per_second = kernels_per_second / batch_size

    # now load in our existing neural networks
    model1 = load_model('/home/vasileios.skliris/ml-validation/HLV_NET/Run_13/elevatedVirgo/model1_32V_No5.h5')

    # let's make sure we're starting with a fresh repo
    repo_path = str(Path.home() / "testhermes")
    try:
        shutil.rmtree(repo_path)
    except FileNotFoundError:
        pass
    repo = qv.ModelRepository(repo_path)

    # # make a dummy model that we'll use to pass the strain

    model = repo.add("my_model", platform=qv.Platform.ENSEMBLE) #ENSEMBLE MIGHT NEED TO CHANGE


    model.export_version(
        model1,
        input_shapes = {"input_type": model.input_shape} 
        output_names = ["output"])

        #input_shapes={"hoft": (-1, NUM_IFOS, kernel_size)},
        #output_names=["prob"]

    class Callback:
        def __init__(self, num_inferences, batch_size):
            self.batch_size = batch_size
            self.y = np.zeros((num_inferences * batch_size,))
            self._i = 0

        def __call__(self, response, request_id, sequence_id):
            start = request_id * self.batch_size
            stop = start + len(response)
            self.y[start: stop] = response[:, 1]
            self._i += 1

            if stop + 1 >= len(self.y):
                return self.y

        def block(self, i: int) -> None:
            while self._i <= i:
                time.sleep(1e-3)


    callback = Callback(num_inferences, batch_size)

    # set up a new directory just for our metrics
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(exist_ok=True)
    metrics_file = metrics_dir / "non-streaming_single-model.csv"

    # reset CUDA_VISIBLE_DEVICES so that we can expose the
    # correct GPUs to Triton
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices


    
    #condor sumbit files to open servers ...
    for s in [1]:# in numer_of_servers:
        submit_job = os.popen("python run-triton-server.py",'r')

    time.sleep(120)

    for job_id in job_ids:
        is_server_ready
    trials =  0
    while condor job is not running)

        wait 1

        trials+=1
        if trials > 10:
            raise("Job didn't run")

    ip = get the ip of the server()

    client = InferenceClient(
        ip,
        model_name="my_model",
        model_version=1,
        batch_size=batch_size,
        callback=callback
    )
        

    monitor = ServerMonitor(
        model_name="my_model",
        ips=ip,
        filename=metrics_file,
        model_version=1,
        name="monitor",
        rate=4
    )
    

    with client, monitor:
        for i in range(int(num_inferences)):
            print(i,'/',num_inferences)
            start = i * inference_stride * batch_size
            stop = start + kernel_size * batch_size
                
            kernel = {'input_data' : data[ start : stop ]}            
#             kernel = {
#                 "strain": hoft1[start: stop].reshape(batch_size, -1, NUM_IFOS),
#             }
            

            client.infer(kernel, request_id=i)

            # block the first few requests to let TensorFlow warm up
            if i < 5:
                callback.block(i)

            # sleep to roughly maintain our inference rate
            time.sleep(0.9 / batches_per_second)

            while True:
                results = client.get()
                if results is not None:
                    break
                time.sleep(1e-3)
            
            output_queue.add(results)
            
