# KDD-2025-FedDiAL

This repository provides the design and implementation details for our proposed method, FedDiAL.

# Prerequisites

- Ubuntu 20.04
- Python 3.5+
- PyTorch
- CUDA environment

# Directory Structure

- ./FedDiAL.py: Contains configuration settings and the basic framework for Federated Learning.
- ./Sim.py: Describes simulators for clients and the central server.
- ./Utils.py: Includes necessary functions and provides guidance on obtaining training and testing data.
- ./Settings.py: Specifies the required packages and settings.
- ./Functions.py: Contains the code for our specific designs.

# Implementation

- To execute the algorithms, run the ./FedDiAL.py file using the following command:

```
   python ./FedDiAL.py
```

- Adjust the parameters and configurations within the ``FedDiAL.py'' file to suit your specific needs.


# Citation

If you use the simulator or some results in our paper for a published project, please cite our work by using the following bibtex entry

```
@inproceedings{yan2025feddial,
  title={FedDiAL: Adaptive Federated Learning with Hierarchical Discriminative Network for Large Pre-trained Models},
  author={Gang Yan, Wan Du},
  booktitle={Proc. of ACM SIGKDD},
  year={2024}
}
```
