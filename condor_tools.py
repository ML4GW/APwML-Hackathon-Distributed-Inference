import os


def is_server_ready(job_id):
    '''
    Returns True if the condor job with the given job_id is running,
    returns False if not. 
    
    '''
    
    if not type(job_id) is str:
        job_id = str(job_id)
    condor_q = os.popen(f"condor_q {job_id}").read()
    query = [line for line in condor_q.split("\n") if "Total for query" in line][0]
    status = False
    if "running" in query and "0 running" not in query:
        status = True
    return status


## tests 
# Example 1
job_id = '111007034.0'
is_server_ready(job_id)

job_id = '111353228.0'
is_server_ready(job_id)

job_id = '111685669.0'
is_server_ready(job_id)







