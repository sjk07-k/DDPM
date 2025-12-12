# DDPM
Denoising Diffusion Probabilistic Models on PathMNIST
Overview

This repository contains an end-to-end implementation of a Denoising Diffusion Probabilistic Model (DDPM) trained on the PathMNIST medical imaging dataset. The project demonstrates the core principles of diffusion-based generative modelling using a time-conditioned U-Net architecture and provides a concise, reproducible tutorial suitable for academic coursework.

The focus of this work is conceptual clarity rather than state-of-the-art performance. The model is intentionally compact and trained for a small number of epochs in order to clearly illustrate the mechanics of forward diffusion, reverse denoising, and iterative sample generation.

Repository Structure
DDMP.ipynb
Denoising Diffusion Probabilistic Models for Medical Image Generation.pdf
LICENSE
README.md

Dataset

The model is trained on PathMNIST, part of the MedMNIST benchmark collection. PathMNIST consists of 28×28 RGB histopathological image patches derived from colorectal tissue slides.

Original source: https://medmnist.com/

Images are normalised to the range [−1, 1]

Class labels are ignored, and the dataset is treated as unconditional

For convenience and transparency, the dataset has been exported to CSV format. Each row in the CSV corresponds to a flattened image, with pixel intensities stored as numerical values.

Model Description

The generative model follows the Denoising Diffusion Probabilistic Model (DDPM) framework:

Forward process: fixed Gaussian noising with a linear variance schedule

Reverse process: learned denoising using a neural network

Architecture: compact U-Net with sinusoidal timestep embeddings

Training objective: mean squared error between true and predicted noise

The network learns to approximate the reverse diffusion distribution by predicting the noise added at each timestep.

Training Configuration

Diffusion steps: T = 200

Noise schedule: linear, 
𝛽
𝑡
∈
[
10
−
4
,
0.02
]
β
t
	​

∈[10
−4
,0.02]

Optimiser: Adam

Learning rate: 2 × 10⁻⁴

Batch size: 128

Epochs: 2

Loss function: Mean Squared Error (MSE)

A random diffusion timestep is sampled at each iteration, and the model is trained to predict the corresponding noise component.

Results

The repository includes visualisations demonstrating:

Training loss convergence

Forward diffusion from clean images to noise

Synthetic image samples generated from pure Gaussian noise

Although trained for only two epochs, the model captures non-trivial colour distributions and texture patterns characteristic of histopathological images.

Ethical Considerations

This project uses medical imaging data for research and educational purposes only. Generated samples are synthetic and must not be interpreted as real clinical data or used for diagnostic decision-making.
Requirements

To run the notebook locally or in Google Colab:

pip install torch torchvision medmnist matplotlib numpy


A GPU is optional but recommended for faster training.

How to Run

Open DDMP.ipynb in Jupyter or Google Colab

Run all cells sequentially

Generated images and plots will be displayed and can be saved for reporting

References

Key references include:

Ho, J., Jain, A. and Abbeel, P. (2020) Denoising diffusion probabilistic models.

Nichol, A.Q. and Dhariwal, P. (2021) Improved denoising diffusion probabilistic models.

Yang, J. et al. (2021) MedMNIST: A lightweight benchmark for medical image classification.

Ronneberger, O., Fischer, P. and Brox, T. (2015) U-Net: Convolutional networks for biomedical image segmentation.

See the full reference list in the accompanying report.

Licence

This project is released under the MIT License.
You are free to use, modify, and distribute the code for educational and research purposes, provided that appropriate credit is given.

Potential ethical concerns include dataset bias, misuse of synthetic medical imagery, and misinterpretation of generated samples. Responsible use requires transparency, clear documentation, and domain expert oversight.
