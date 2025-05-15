
# 🧠 Simple Models, Rich Representations: Visual Decoding from Primate Intracortical Neural Signals

## 📄 Abstract

Understanding how patterns of neural activity give rise to perception remains a central challenge in neuroscience. Here, we tackle the problem of decoding visual information from high-density intracortical recordings in primates. Analyzing the THINGS Ventral Stream Spiking Dataset, we systematically explore the impact of model architecture, objective function, and data scaling on decoding performance.
Our findings reveal that the decoding accuracy is primarily driven by modeling temporal dynamics in neural activity, rather than architectural complexity. We show that a simple model combining temporal attention with a shallow MLP achieves top-1 image retrieval accuracy up to around 70%, outperforming linear baselines and recurrent or convolutional models.
Scaling experiments reveal predictable regimes of diminishing returns with respect to input dimensionality and training set size. Leveraging these insights, we propose a modular generative decoding pipeline that integrates low-resolution latent reconstruction with semantically conditioned diffusion. Following recent trends in increasing inference-time computation, our approach generates multiple candidate outputs conditioned on neural signals, which are subsequently ranked based on their structural similarity to the initial latent reconstruction. This strategy enables the generation of plausible visual reconstructions from 200 ms of brain activity.
Our findings offer guiding principles for future brain-computer interfaces and open new avenues for semantic brain decoding.

---

## 📘 Notebooks Overview

### `monkeys_retrieve.ipynb`: Analysis & Decoding

1. **Image Extraction**
   - Loading and visual inspection of visual stimuli.

2. **Neural Response Collections**
   - Using a 200 ms post-stimulus window, temporally aligned with stimulus onset.
  
3. **Average Activity Over Repetitions**
   - Averaging across stimulus repetitions to reduce noise.

4. **Time Neural Network**
   - Simple Temporal attention network to aggregate dynamic neural patterns.

5. **PCA on Channels (Scaling Law)**
   - Dimensionality reduction and scaling behavior inspection over input channels.

6. **Random Subsampling (Scaling Law)**
   - Analysis of training set size impact on decoding performance.

### `monkeys_gener.ipynb`: Generative Decoding Pipeline

1. **Neural Data Preprocessing**
   - Feature preparation from averaged neural signals.

2. **Soft Mapping Model**
   - Light MLP with attention for mapping neural activity to visual and structural latent space.

3. **Structural Decoding**
   - Image generation via Stable Diffusion XL using predicted latents.

4. **Rejection Sampling**
   - Ranking multiple generated images by structural similarity to the predicted latent representation.

5. **Final Reconstruction**
   - Selection of the most plausible reconstruction based on SSIM.

---

## 📊 Key Results & Visualizations

- 🔥 **Attention Heatmaps**  
  - Visual attention across timepoints reveals model focus during decoding.

- 🧠 **Latent Reconstructions**  
  - Predicted latent vectors compared with true latents, visualized via SSIM or cosine similarity.

- 🖼️ **Generated Images**  
  - Output of the generative pipeline using Stable Diffusion XL, with visual comparison to target or retrieved images.

---

## ✅ Conclusion

This pipeline demonstrates that:

- Simple attention-based models can outperform more complex architectures in neural decoding.
- Modeling temporal structure in neural activity is key to decoding accuracy.
- A modular generative decoder with latent reconstruction and semantic ranking produces plausible image reconstructions.
