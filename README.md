# Federated Learning on Non-IID and Globally Long-Tailed Data via Aggregated Re-Weighting Networks

This is the code for paper :  **Federated Learning on Non-IID and Globally Long-Tailed Data via Aggregated Re-weighting Network**.

**Abstract**: One major problem concerned in federated learning is data non-IIDness. Because the training samples are collected and stored locally on each client's device, the machine learning procedure does not meet the requirement of independent and identical distribution (IID). Existing federated learning methods to deal with non-IID data generally assume that the data is globally balanced. However, real-world multi-class data tends to exhibit long-tail distribution, where the majority of samples are in few head classes and a large number of tail classes only have a small amount of data. This paper therefore focuses on addressing the problem of handling non-IID and globally long-tailed data in federated learning scenario. Accordingly, we propose a new federated learning method called Federated Aggregated Re-weighting Networks (FedARN). It assigns different weights to the local training samples by re-weighting networks, which are trained to learn a proper loss-weight mapping by meta-learning techniques. To deal with data non-IIDness and global long-tail, the re-weighting networks are aggregated on the server to implicitly acquire the knowledge of global long-tail distribution. We further propose an asynchronous meta updating mechanism to reduce the communication cost for meta-learning training. Experiments on several long-tailed image classification benchmarks show that FedARN outperforms the state-of-the-art federated learning methods.

## Dependencies

* PyTorch >= 1.0.0

* torchvision >= 0.2.1

  

## Parameters

| Parameter     | Description                                              |
| ------------- | -------------------------------------------------------- |
| `dataset`     | Dataset to use. Options: `cifar10`,`cifar100`, `fmnist`. |
| `lr`          | Learning rate of model.                                  |
| `v_lr`        | Learning rate of re-weighting network.                   |
| `local_bs`    | Local batch size of training.                            |
| `test_bs`     | Test batch size .                                        |
| `num_users`   | Number of clients.                                       |
| `frac`        | the fraction of clients to be sampled in each round.     |
| `epochs`      | Number of communication rounds.                          |
| `local_ep`    | Number of local epochs.                                  |
| `imb_factor`  | Imbalanced control. Options: `0.01`,`0.02`, `0.1`.       |
| `num_classes` | Number of classes.                                       |
| `num_meta`    | Number of meta data per class.                           |
| `device`      | Specify the device to run the program.                   |
| `seed`        | The initial seed.                                        |


## Usage

Here is an example to run FedARN on CIFAR-10 with imb_fartor=0.01:

```
python main.py --dataset=cifar10 \
    --lr=0.01 \
    --v_lr=0.01\
    --epochs=500\
    --local_ep=5 \
    --num_users=20 \
    --num_meta=10 \
    --num_classes=10 \
    --imb_factor=0.01\
```

