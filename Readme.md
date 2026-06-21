# Anime Face Generator — DCGAN

Generate anime-style faces from random noise using a Deep Convolutional Generative Adversarial Network (DCGAN) built with TensorFlow/Keras, trained on the [Anime Faces](https://www.kaggle.com/datasets/soumikrakshit/anime-faces) dataset.

## Overview

This project implements a GAN from scratch to synthesize 64x64 anime face images. It includes two model versions:

- **v1 (baseline)** — A standard DCGAN generator/discriminator pair, trained with the Adam optimizer at a learning rate of `1e-4`.
- **v2 (improved)** — A deeper generator (starting from a `8x8x512` projection instead of `8x8x256`) with an extra upsampling block, trained using the DCGAN-recommended optimizer settings (`lr=0.0002`, `beta_1=0.5`) for more stable convergence.

Both models follow the classic adversarial setup: the **Generator** learns to map a random noise vector into a realistic-looking face, while the **Discriminator** learns to distinguish real dataset images from generated ones. The two networks are trained jointly in a minimax game.

## Demo

After training, the generator can produce novel anime faces from random latent vectors:

```python
seed = tf.random.normal([16, 100])  # or NOISE_DIM for v2
generate_and_save_images(generator, epoch)
```

## Dataset

- **Source:** [soumikrakshit/anime-faces](https://www.kaggle.com/datasets/soumikrakshit/anime-faces) on Kaggle (~21,500 images)
- **Preprocessing:**
  - Resized to `64x64x3`
  - Converted from BGR to RGB
  - Normalized to the range `[-1, 1]`
  - Batched and shuffled using `tf.data.Dataset`

```python
import kagglehub
path = kagglehub.dataset_download("soumikrakshit/anime-faces")
```

## Model Architecture

### Generator (v1)
```
Dense(8*8*256) → BatchNorm → LeakyReLU → Reshape(8,8,256)
→ Conv2DTranspose(128) → BatchNorm → LeakyReLU
→ Conv2DTranspose(64)  → BatchNorm → LeakyReLU
→ Conv2DTranspose(3, activation='tanh')   # 64x64x3 output
```

### Generator (v2 — deeper)
```
Dense(8*8*512) → BatchNorm → LeakyReLU → Reshape(8,8,512)
→ Conv2DTranspose(256) → BatchNorm → LeakyReLU
→ Conv2DTranspose(128) → BatchNorm → LeakyReLU
→ Conv2DTranspose(64)  → BatchNorm → LeakyReLU
→ Conv2DTranspose(3, stride=1, activation='tanh')  # 64x64x3 output
```

### Discriminator
```
Conv2D(64, stride=2)  → LeakyReLU → Dropout(0.3)
Conv2D(128, stride=2) → LeakyReLU → Dropout(0.3)
Flatten → Dense(1)   # real/fake logit
```

## Training Details

| Setting | v1 | v2 |
|---|---|---|
| Noise dimension | 100 | 128 |
| Optimizer | Adam | Adam |
| Learning rate | 1e-4 | 2e-4 |
| Beta 1 | default | 0.5 |
| Loss | Binary Cross-Entropy (from logits) | Binary Cross-Entropy (from logits) |
| Batch size | 128 | 128 |
| Image size | 64x64x3 | 64x64x3 |

Loss functions:
- **Discriminator loss** = BCE(real, ones) + BCE(fake, zeros)
- **Generator loss** = BCE(fake, ones) — the generator is rewarded for fooling the discriminator.

Training uses `@tf.function`-compiled steps for GPU-accelerated execution, with gradients computed via `tf.GradientTape` for both networks at each step.

## Requirements

```
tensorflow>=2.x
numpy
opencv-python
matplotlib
kagglehub
```

Install with:

```bash
pip install tensorflow numpy opencv-python matplotlib kagglehub
```

## Usage

1. **Download the dataset**
   ```python
   import kagglehub
   path = kagglehub.dataset_download("soumikrakshit/anime-faces")
   ```
2. **Preprocess images** into a normalized NumPy array and wrap in a `tf.data.Dataset`.
3. **Build the models**
   ```python
   generator = build_generator()
   discriminator = build_discriminator()
   ```
4. **Train**
   ```python
   for epoch in range(EPOCHS):
       for image_batch in train_dataset:
           g_loss, d_loss = train_step(image_batch)
       generate_and_save_images(generator, epoch + 1)
   ```
5. **Generate new faces** by sampling random noise and passing it through the trained generator.

## Project Structure

```
.
├── anime_face_gan.ipynb   # Main notebook (data prep, models, training loop)
└── README.md
```

## Notes & Future Improvements

- Increase training epochs and add checkpointing (`tf.train.Checkpoint`) to resume long runs.
- Track losses with TensorBoard for easier convergence monitoring.
- Experiment with spectral normalization or WGAN-GP loss for improved training stability.
- Try progressively growing resolution (e.g., 64 → 128 → 256) for higher-fidelity outputs.

## Acknowledgements


- Architecture inspired by the original [DCGAN paper](https://arxiv.org/abs/1511.06434) and the [TensorFlow DCGAN tutorial](https://www.tensorflow.org/tutorials/generative/dcgan).

## License

This project is released under the MIT License, allowing anyone to use, modify, and distribute the code with proper attribution. See the [LICENSE](LICENSE) file for more information.
