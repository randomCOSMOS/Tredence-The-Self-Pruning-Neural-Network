# Self-Pruning Neural Network Report

## Overview

This project implements a self-pruning neural network for CIFAR-10 image classification.  
Instead of pruning weights only after training, the model learns a gate for every prunable weight during training itself.  
These gates decide which connections remain active and which ones are effectively removed.

The final model was trained and evaluated for three different values of `lambda`, which controls the strength of the sparsity penalty.

## Why an L1 Penalty on Sigmoid Gates Encourages Sparsity

Each prunable weight has a learnable gate score. During the forward pass, that score is passed through a sigmoid:

`gate = sigmoid(gate_score)`

This keeps every gate value between 0 and 1. The effective weight used by the layer is:

`effective_weight = weight * gate`

If a gate becomes very small, the corresponding weight contributes almost nothing to the output.  
To encourage this behavior, the training objective adds an L1 penalty on the gate values:

`Total Loss = Classification Loss + lambda * Sparsity Loss`

where the sparsity loss is computed from the gate values of all `PrunableLinear` layers.

L1 regularization is well known for promoting sparsity because it applies a direct penalty to non-zero values.  
In this case, it pushes unnecessary gates toward 0. Once a gate falls below the pruning threshold of `0.01`, that weight is counted as pruned and is zeroed during hard pruning.

## Results for Different Lambda Values

The table below summarizes the final test accuracy and sparsity level for the three lambda values tested:

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
| --- | ---: | ---: |
| 0.05 | 82.78 | 42.56 |
| 0.10 | 83.04 | 64.20 |
| 0.20 | 82.72 | 77.10 |

## Best Model Analysis

The best overall trade-off was obtained with **lambda = 0.10**.

- Test Accuracy: **83.04%**
- Sparsity Level: **64.20%**

This result is strong because the network keeps competitive CIFAR-10 accuracy while pruning nearly two-thirds of the prunable weights.

## Gate Value Distribution for the Best Model

The plot below shows the distribution of the final gate values for the best model:

![Best gate distribution](gate_distribution_best.png)

This distribution is exactly the pattern expected from a successful self-pruning network:

- There is a **large spike near 0**, which corresponds to weights that were effectively pruned.
- There is another **cluster of values away from 0, near 1**, which corresponds to important weights that stayed active.

This separation shows that the model did not simply shrink all gates uniformly.  
Instead, it learned to keep a subset of useful connections open while pushing less important ones close to zero.

## Trade-Off Summary

The lambda sweep clearly shows the expected sparsity-versus-accuracy trade-off:

- A smaller lambda (`0.05`) preserves more connections, resulting in lower sparsity but strong accuracy.
- A medium lambda (`0.10`) gives the best balance between pruning and performance.
- A larger lambda (`0.20`) pushes more gates toward zero, increasing sparsity further but causing a small drop in test accuracy.

Overall, the experiments show that the model is successfully pruning itself during training, and that the pruning behavior can be controlled through the lambda hyperparameter.
