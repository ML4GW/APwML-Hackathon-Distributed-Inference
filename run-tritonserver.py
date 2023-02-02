#!/usr/bin/python3

import os
import numpy as np
from pycondor import Job
import subprocess
import time

def fancyname(directory):

    date = time.strftime('%Y%m%d')

    dirs =  os.listdir(directory)

    today_dirs = list( dir_ for dir_ in dirs if date in dir_)
    
    if today_dirs:
        indeces = list(int(dir_.split("_")[-1]) for dir_ in today_dirs)
    
        index_ = str(max(indeces)+1)
    else:
        index_="1"    

    return date+"_"+index_

def main (model_repo = "/home/mly/O3-replay/.testhermes"
         , condor_path="./"
         , name = "run-tritonserver"
         , number_of_servers=1):

        
    fancy_name = fancyname(condor_path)
    
    ip_list=[]

    for s in range(number_of_servers):

        server_no = s+1

        error = condor_path+"server-submition_"+fancy_name+"/error"
        output = condor_path+"server-submition_"+fancy_name+"/output"
        log = condor_path+"server-submition_"+fancy_name+"/log"
        submit = condor_path+"server-submition_"+fancy_name+"/submit"

        job = Job(name=name+"_"+str(server_no),
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

        expression = "cat " + log + "/" + name+"_"+str(server_no)+".log" +  " | grep \"Job executing on host\"| tail -n1 " + "| sed -r \"s/.*Job\Wexecuting\Won\Whost:\W<(.*)\?addrs.*/\\1/\""

        print(expression)

        ip = subprocess.check_output(expression , shell=True )
        while str(ip)=="b''":
            time.sleep(1)
            ip = subprocess.check_output(expression , shell=True )
        
        print(str(ip)[2:-3])
        ip = str(ip)[3:-3]

        print(ip)
        ip_list.append(ip)

    print(ip_list)
    return(ip_list)




if __name__ == "__main__":
    main()


