#!/usr/bin/python3

import os
from pycondor import Job, Dagman

# apptainer run /cvmfs/singularity.opensciencegrid.org/ml4gw/hermes/tritonserver:22.12

error = 'condor/error'
output = 'condor/output'
log = 'condor/log'
submit = 'condor/submit'


job = Job(name="run-tritonserver",
          executable="/opt/tritonserver/bin/tritonserver",
          submit=submit,
          error=error,
          output=output,
          log=log,
          requirements="HasSingularity",

          
          extra_lines=["accounting_group_user="+os.environ['HOME'].split("/")[-1],
                       "accounting_group=ligo.dev.o4.burst.allsky.mlyonline",
                       "request_disk=64M",
                       "request_GPUs = 1",
                       "request_cpus = 8",
                       "transfer_executable = False",
                      # "HasSingularity=True",
                       #"SingularityVersion=\"/cvmfs/singularity.opensciencegrid.org/ml4gw/hermes/tritonserver:22.12\""])
                      "+SingularityImage = \"/cvmfs/singularity.opensciencegrid.org/ml4gw/hermes/tritonserver:22.12\""]
          )

# job = Job(name="run-tritonser
#           requirements="'HasSingularity'",
#           executable="/opt/tritonserver/bin/tritonserver",
#         #   executable="apptainer run /cvmfs/singularity.opensciencegrid.org/ml4gw/hermes/tritonserver:22.12",
#           error=condor_dir,
#           log=condor_dir,
#           output=condor_dir,
#           extra_lines=["accounting_group_user="+os.environ['HOME'].split("/")[-1],
#                        "accounting_group=allsky.mlyonline",
#                        "request_disk=64M",
#                        "+SingularityImage=/cvmfs/singularity.opensciencegrid.org/ml4gw/hermes/tritonserver:22.12"])


job.build_submit(fancyname=True)


