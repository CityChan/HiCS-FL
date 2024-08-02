# Accelerating Non-IID Federated Learning via Heterogeneity-Guided Client Sampling

This is the code for the NeurIPS2023 submission


## Install packages in requrement.txt

`pip install -r requirements.txt`

We use pytorch 1.9.0 + rocm 4.2 environment with a python version 3.6.13. All requried packages are in requirements.txt.


## Code instruction
- `models.py`: the model's structure used in the experiments
- `utils.py`: utilization functions for computing metrics of the experiments
- `DivFL_utils.py`: utilization functions for DivFL sampling method
- `clustering_utils.py`: utilization function for Clustered sampling method
- `sampling.py`: functions for generating data partitions with Dirichlet distribution
- `HiCS.py`: utilization function for HiCS-FL sampling method
- `train.py`: training main function

## Running an experiment

- --dataset: CIFAR10, CIFAR100, FMNIST
- --batch_size: size of mini batch
- --num_epochs: total number of global communication rounds
- --num_clients: number of clients
- --sampling_rate: fraction of clients participating local training each global round
- --local_ep: number of local epochs
- --alphas: list of concentration parameters for generating data partitions
- --T: scaling parameter, temperature
- --seed: random seed for generating data partitions
- --alg: random, pow-d, CS, DivFL, HiCS
- --lr: initializing learning rate

We gave an example in `train_script.sh`









