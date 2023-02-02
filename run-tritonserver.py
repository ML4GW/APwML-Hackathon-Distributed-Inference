#!/usr/bin/python3

import os
import numpy as np
from pycondor import Job
import subprocess



def main (model_repo = "/home/mly/O3-replay/.testhermes"
             , condor_path="."
             , condor_file_sufix = None 
             , name = "condor_run-tritonserver"):

    if condor_file_sufix is None:
        condor_file_sufix = str(np.random.randint(1,1000))

    error = condor_path+"/"+name+"_"+condor_file_sufix+"/error"
    output = condor_path+"/"+name+"_"+condor_file_sufix+"/output"
    log = condor_path+"/"+name+"_"+condor_file_sufix+"/log"
    submit = condor_path+"/"+name+"_"+condor_file_sufix+"/submit"

    job = Job(name=name,
            executable="/opt/tritonserver/bin/tritonserver",
            submit=submit,
            error=error,
            output=output,
            log=log,
            requirements="HasSingularity && (CUDACapability >= 6.0)",
            
            extra_lines=["accounting_group_user="+os.environ['HOME'].split("/")[-1],
                        "accounting_group=ligo.dev.o4.burst.allsky.mlyonline",
                        "request_disk=64M",
                        "request_GPUs = 1",
                        "request_cpus = 8",
                        "transfer_executable = False",
                        "+SingularityImage = \"/cvmfs/singularity.opensciencegrid.org/ml4gw/hermes/tritonserver:22.12\""]
            )

    job.add_arg("--model-repository /home/mly/O3-replay/.testhermes")
    # var=$(echo -e $(python run-tritonserver.py) | grep "submitted" | awk '{pint $8}' | tr -d '.')
    job.build_submit(fancyname=False)

    expression = "cat " + log + "/" + name + ".log" + " | grep \"Job executing on host\" | tail -n1 " + "| sed -r \"s/.*Job\Wexecuting\Won\Whost:\W<(.*)\?addrs.*/\\1/\""
    
    print(expression)

    ip = subprocess.check_output(expression , shell=True )

    print(ip)
    return(ip)



if __name__ == "__main__":
    p = main()
    print(p)


