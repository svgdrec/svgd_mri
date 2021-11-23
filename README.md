# Posterior Sampling and Uncertainty in Accelerated MRI using Stein Variational Gradient Descent
Supplementary material for CVPR 2022 submission.

![title gif](./results/title.gif?raw=true "")

## Structure of repository
1. **code** - Self-contained code repository of the project. Requires to download NYU fastMRI dataset and weights.
2. **ex_results** - Supplementary results of sampling. Containing GIFs mentioned in the paper.

## Abstract
*Image reconstruction methods for undersampled MR acquisitions focus on reconstructing a single image, ignoring the possibility of multiple plausible reconstructions due to the missing data. Though several recent works were proposed addressing this inversion uncertainty by sampling multiple images, these either came with low image quality, poor data consistency or long inference times. In this work, we propose a method based on Stein Variational Gradient Descent (SVGD) for sampling from the posterior distribution of images given the measured data. As the SVGD requires gradients of the target posterior, we approximate these using the intermediate steps of a state-of-the-art unrolled reconstruction algorithm, the end-to-end VarNet. Furthermore, we calculate the kernel distances for the SVGD in the latent space of an auto-encoder, which leads to structural diversity in the samples. We evaluate the method on the fastMRI dataset and show that the method yields high reconstruction quality with diversity in samples with a negligible additional computational load. Comparisons reveal that SVGD samples show high variability where the VarNet produces hallucinations. In contrast to the compared methods, the proposed method shows higher structural diversity in the phase encoding direction where the main uncertainty is expected in an undersampled acquisition.*
