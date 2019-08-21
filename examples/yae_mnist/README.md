## Catalyst.DL – Y-Autoencoder implementation on MNIST

Original paper https://arxiv.org/pdf/1907.10949.pdf

### Local run

```bash
catalyst-dl run --config=./yae_mnist/config.yml
```

### Training visualization

For tensorboard visualization use 

```bash
CUDA_VISIBLE_DEVICE="" tensorboard --logdir=./logs/yae_mnist
```
