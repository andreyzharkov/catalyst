## Catalyst.DL – cifar10 Y-AE example

https://arxiv.org/pdf/1907.10949.pdf

### Local run

```bash
catalyst-dl run --config=./autoencoders/config.yml
```

### Training visualization

For tensorboard visualization use 

```bash
CUDA_VISIBLE_DEVICE="" tensorboard --logdir=./logs/autoencoders
```
