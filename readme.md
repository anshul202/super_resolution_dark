instructions to be considered;

Based on the current state-of-the-art research for joint low-light enhancement and super-resolution, both the dataset and model classes in the barebones code need significant advancements to achieve high-quality results.

Here are the specific parts you should work on advancing:

**1. Advancing the `DarkSRDataset` Class**
The current dataset class loads the entire image into memory, which will quickly cause Out-Of-Memory (OOM) errors on a Google Colab GPU and lacks the robustness needed for deep learning. You should advance this class by adding:

* **Dynamic Chunking/Cropping:** Instead of returning the full image, modify the `__getitem__` method to randomly crop the arrays into smaller chunks (e.g., $256 \times 256$ or $512 \times 512$ patches) before converting them to tensors.


* **Data Augmentation:** Implement geometric augmentations, specifically random horizontal and vertical flipping of the cropped patches, to prevent the network from overfitting.


* **Synthetic Degradation (Optional but recommended):** To make the model more robust, you can add code to randomly apply gamma correction ($\gamma \in [2.0, 5.0]$) and inject synthetic Gaussian or Poisson noise to the input images during loading.



**2. Advancing the `HybridTransformerCNN` Class**
The barebones model uses standard PyTorch layers that are too generic for this highly ill-posed problem. You should upgrade the three main blocks of the architecture:

* **The CNN Frontend:** Replace the basic `nn.Conv2d` layers with Residual Channel Attention Groups (RCAGs) or Semantic-Aligned Scale-Aware Modules (SAM). These advanced convolutional blocks are much better at isolating high-frequency sensor noise without blurring the underlying structural edges.


* **The Transformer Bottleneck:** The standard `nn.TransformerEncoder` treats all pixels equally, which is inefficient. You should replace this with a more specialized attention mechanism. For example, you could implement a Swin Transformer block to utilize hierarchical window-based attention , or use a Structure-Guided Transformer Block (SGTB) that specifically leverages structural cues to preserve details. To handle complex non-uniform lighting, integrating a quaternion illumination estimation module into the transformer can help disentangle lighting variations.


* **Frequency Domain Separation:** Before passing the CNN features to the Transformer, consider implementing a Discrete Wavelet Transform (DWT). This separates the image data into low-frequency components (which represent the dark illumination) and high-frequency components (which represent the textures), allowing the transformer to process the lighting shifts without accidentally hallucinating incorrect textures.



**3. Advancing the Training Loop (Loss Functions)**
While not a class itself, the `nn.L1Loss()` in the training loop must be upgraded. State-of-the-art models rely on a weighted composite loss function to balance brightness, structure, and realism. You should calculate and sum three distinct losses during each batch:

* **Charbonnier Loss:** Use this instead of L1 or MSE, as it is significantly more robust to the outliers caused by severe low-light noise.


* **Multi-Scale SSIM Loss:** Add this to strictly enforce structural alignment and preserve the physical shapes of objects in the dark.


* **Perceptual (VGG) Loss:** Pass your output and the ground truth through a pre-trained VGG-16 or VGG-19 network and compare their feature maps. This ensures the final super-resolved image looks perceptually sharp and natural to the human eye.


Just somehow make this shit work...