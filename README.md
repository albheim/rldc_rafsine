# DC-sim

This is a toolbox for simulating the heat loop in a small datacenter. It has two backends, first a very simple one described [here](link_paper) and then 
a more complex CFD based one described [here]()

## Installation
Clone this repo to you local drive and install the packages defined in `requirements.txt` using any desired method.
```
pip install -f requirements.txt
``` 

## Ray setup 
First you start the ray daemon depending on your cluster. For a local one-node cluster this is done with
```
ray start --head --num-cpus=X
```
where it can be nice to not use all cpus for ray so that tensorboard can work in the background.

For a cluster with multiple nodes there are resources on the ray website.

## Run the training
Finally the simulation is run with 
```python
python3 main.py
```
where there are some optional parameters available
```
  --rafsine # Use rafsine instead of simple simulation, needs additional git repo
  --actions [ACTIONS ...] # List what actions the agent should take
  --observations [OBSERVATIONS ...] # List what observations the agent should get
  --n_samples # How many samples the hyperparameter search should run over
  --tag # Name of the run
  --stop_timesteps # How many seconds (in simulation) should each sample run
```

## Analyse
Tensorboard is started with 
```
tensorboard --logdir /path/to/repo/results --reload_multifile true
```
and can be accessed with a browser at `localhost:6006`.

Ray dashboard is running with the ray deamon and can be access with a browser at `localhost:8265`.