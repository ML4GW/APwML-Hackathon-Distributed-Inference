#!/usr/bin/python3

import os
from pycondor import Job

# apptainer run /cvmfs/singularity.opensciencegrid.org/ml4gw/hermes/tritonserver:22.12

condor_dir = "condor_run-tritonserver/"

job = Job(name="run-tritonserver",
          requirements="HasSingularity",
        #   executable="/opt/tritonserver/bin/tritonserver",
          executable="apptainer run /cvmfs/singularity.opensciencegrid.org/ml4gw/hermes/tritonserver:22.12",
          submit=condor_dir,
          error=condor_dir,
          log=condor_dir,
          output=condor_dir,
          extra_lines=["accounting_group_user="+os.environ['HOME'].split("/")[-1],
                       "accounting_group=allsky.mlyonline",
                       "request_disk=64"])
                    #    "+SingularityImage=/cvmfs/singularity.opensciencegrid.org/ml4gw/hermes/tritonserver:22.12"])

job.build_submit()