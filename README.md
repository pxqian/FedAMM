# LONG-TAILED FEDERATED LEARNING VIA AGGREGATED META MAPPING

This is the code for paper :  **LONG-TAILED FEDERATED LEARNING VIA AGGREGATED META MAPPING**.

**Abstract**: One major problem concerned in federated learning is data non-IIDness. Existing federated learning methods to deal with non-IID data generally assume that the data is globally balanced. However, real-world multi-class data tends to exhibit long-tail distribution.
  Therefore, we propose a new federated learning method called Federated Aggregated Meta Mapping (FedAMM) to address the joint problem of non-IID and globally long-tailed data in a federated learning scenario. FedAMM assigns different weights to the local training samples by trainable loss-weight mapping in a meta-learning manner. To deal with data non-IIDness and global long-tail, the meta loss-weight mappings are aggregated on the server to implicitly acquire global long-tail distribution knowledge. We further propose an asynchronous meta updating mechanism to reduce the communication cost for meta-learning training. Experiments on several classification benchmarks show that FedAMM outperforms the state-of-the-art federated learning methods. 

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

