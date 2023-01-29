import argparse
import sys

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

# Input creation
import sys, os
sys.path.append("/home/vasileios.skliris/mly/")
usr=os.getcwd().split('/')[2]
if os.path.exists('/home/'+usr+'/mly'):
    sys.path.append('/home/'+usr+'/mly')
    #savepath='/home/'+usr+'/public_html/'
else:
    sys.path.append('/home/vasileios.skliris/mly/')
    #savepath='./'
    
from mly.tools import dirlist
    
# Data assemble
from mly.offlinefar import assembleDataSet, testModel

out = assembleDataSet( masterDirectory = '/home/vasileios.skliris/mlyPipeline/masterdir/'
        , dataSets = ['1345996507-1345996866_360.pkl']
        , detectors = 'HLV'
        , batches = 2
        , batchNumber = 1
        , lags=4
        , includeZeroLag=False)


# model parameters
NUM_IFOS = 3  # number of interferometers analyzed by our model
SAMPLE_RATE = 1  # rate at which input data to the model is sampled
KERNEL_LENGTH = 1  # length of the input to the model in seconds

# inference parameters
INFERENCE_DATA_LENGTH = len(out) # 8192 * 16  # amount of data to analyze at inference time
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

C_kernel_size = int(C_SAMPLE_RATE * C_KERNEL_LENGTH)
C_inference_stride = int(C_SAMPLE_RATE / INFERENCE_SAMPLING_RATE)
C_inference_data_size = int(C_SAMPLE_RATE * INFERENCE_DATA_LENGTH)

# define some parameters which apply to both data streams
num_kernels = inference_data_size // inference_stride
num_inferences = num_kernels // batch_size
kernels_per_second = INFERENCE_RATE * INFERENCE_SAMPLING_RATE
batches_per_second = kernels_per_second / batch_size

# now load in our existing neural networks
model1 = load_model('/home/vasileios.skliris/ml-validation/HLV_NET/Run_13/elevatedVirgo/model1_32V_No5.h5')
model2 = load_model('/home/vasileios.skliris/ml-validation/HLV_NET/Run_13/elevatedVirgo/model2_32V_No6.h5')

# let's make sure we're starting with a fresh repo
repo_path = str(Path.home() / "testhermes")
try:
    shutil.rmtree(repo_path)
except FileNotFoundError:
    pass
repo = qv.ModelRepository(repo_path)

# make a dummy model that we'll use to pass the strain
# input tensor to both models
input_s = tf.keras.Input(name="strain", shape=(1024, 3))
output_s = tf.identity(input_s)
input_model = tf.keras.Model(inputs=input_s, outputs=output_s)

# create another model for the backend of the service
# to combine the outputs from both models
output1 = tf.keras.Input(name="output1", shape=(2,))
output2 = tf.keras.Input(name="output2", shape=(2,))
final_output = output1 * output2
output_model = tf.keras.Model(inputs=[output1, output2], outputs=final_output)

# add all these models to our model repo
qv_model1 = repo.add("model1", platform=qv.Platform.SAVEDMODEL)
qv_model2 = repo.add("model2", platform=qv.Platform.SAVEDMODEL)
qv_input_model = repo.add("input-model", platform=qv.Platform.SAVEDMODEL)
qv_output_model = repo.add("output-model", platform=qv.Platform.SAVEDMODEL)

# add concurrent versions of models 1 and 2 to support our inference rate
qv_model1.config.add_instance_group(count=2)
qv_model2.config.add_instance_group(count=2)

# now export the current versions of these models
# to their corresponding entry in the model repo
qv_model1.export_version(model1)
qv_model2.export_version(model2)
qv_input_model.export_version(input_model)
qv_output_model.export_version(output_model)

# finally, create an ensemble model which will pipe the outputs
# of models into inputs of the next ones in the pipeline
ensemble = repo.add("ensemble", platform=qv.Platform.ENSEMBLE)

# this ensemble will have two inputs, one for the strain data
# and one for the correlation data. The strain data will get fed
# to our "input" model so that we can pipe the output of that
# to the inputs of models 1 and 2
ensemble.add_input(qv_input_model.inputs["strain"])
ensemble.add_input(qv_model2.inputs["correlation"])

# these lines will do the aforementioned routing of the
# strain input model to the inputs of the two NNs
ensemble.pipe(
    qv_input_model.outputs["tf.identity"],
    qv_model1.inputs["strain"],
)
ensemble.pipe(
    qv_input_model.outputs["tf.identity"],
    qv_model2.inputs["strain"],
)

# now route the outputs of these models to the
# input of the output combiner model
ensemble.pipe(
    qv_model1.outputs["main_output"],
    qv_output_model.inputs["output1"],
    key="model1_output",
)
ensemble.pipe(
    qv_model2.outputs["main_output"],
    qv_output_model.inputs["output2"],
    key="model2_output"
)

# finally, expose the output of this combiner model
# as the output of the ensemble
ensemble.add_output(qv_output_model.outputs["tf.math.multiply"])

# export None to indicate that there's no NN we need
# to export, but rather a DAG routing different NNs
# to one another. The path of this DAG is contained
# in the Triton config that hermes has built for you
ensemble.export_version(None)


# callback class which will take the responses from the
# service and insert them into an array in a separate
# callback thread
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


strainList      = out.exportData('strain', shape = (None, 1024, NUM_IFOS)).astype("float32")
correlationList = out.exportData('correlation', shape = (None, 60, NUM_IFOS)).astype("float32")
gpsTimes = out.exportGPS()




# now serve a local Triton instance on GPU 0 using Singularity
with serve(repo_path, gpus=[0]) as instance:

    instance.wait()

    client = InferenceClient(
        "localhost:8001",
        model_name="ensemble",
        model_version=1,
        batch_size=batch_size,
        callback=callback
    )
    monitor = ServerMonitor(
        model_name="ensemble",
        ips="localhost",
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
            c_start = i * C_inference_stride * batch_size
            c_stop = c_start + C_kernel_size * batch_size
            
            kernel = {'strain' : strainList[ start : stop ]
                      ,'correlation' : correlationList[ c_start : c_stop ]}
        
#             kernel = {
#                 "strain": hoft1[start: stop].reshape(batch_size, -1, NUM_IFOS),
#                 "correlation": hoft2[c_start: c_stop].reshape(batch_size, -1, NUM_IFOS)
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
        
    #print(np.expand_dims(results,axis=1).shape, np.array(gpsTimes).shape,np.array(gpsTimes)[:len(results)].shape)
    result=np.hstack((np.expand_dims(results,axis=1),np.array(gpsTimes)[:len(results)]))
    #print(result)
    
    result_pd = pd.DataFrame(result ,columns = ['score']+list('GPS'+str(det) for det in 'HLV'))
    name = 'testFrame'
    with open(name+'.pkl', 'wb') as output:
        pickle.dump(result_pd, output, 4)
        
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

