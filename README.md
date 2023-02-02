# APwML-Hackathon-Distributed-Inference

## Status after hackathon

There are two important tools completed and ready to be modified for more general use.

- run_tritonserver.py creates a triton server using condor, inside a singularity. 
  You can create many servers (input of the function needs some more attention).
  The function returns the ip address of the servers created, to be used by clients.
  
- client_wrapper.py recieves an ip address and a model name already used in the triton servers
  and it creates a client and a queue object. Then it sends inference requests and gets back the
  result, dumped inside the queue.
  
  For model_repo and model data mly models were used.
  
## Possible next steps

- Creating workers that will run the client and the inference requests using condor
