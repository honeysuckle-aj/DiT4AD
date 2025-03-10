# DiT for Anomaly Detection

## Abstract

This study employs a latent denoising diffusion model as a framework in the image reconstruction part, coupled with an improved Vision Transformer
structure as a noise prediction model. This approach achieves more realistic image reconstruction effects and has reached mainstream levels in terms of both FID and sFID
metrics. Benefiting form excellent reconstruction results, I use a CNN model to compare the reconstructed images and the original images and segment out the abnormal
parts. Ultimately, combining reconstruction loss, feature loss and segmentation results as metrics to determine whether the image is abnormal.