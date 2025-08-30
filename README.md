<div align="center">
<h1> ClipMPCAD: Few-Shot Anomaly Detection with LLM-Guided Prompts and Multi-Attention Fusion </h1>
</div>

<div align="center">
This is an official PyTorch implementation of our paper, "Few-Shot Anomaly Detection with LLM-Guided Prompts and Multi-Attention Fusion."
</div>

⭐ Abstract
Medical and industrial image anomaly detection faces challenges like limited data and privacy constraints. While pre-trained large vision-language models (VLMs) show promise, they often suffer from weak cross-modal alignment, insufficient domain-specific supervision, and poor sensitivity to fine-grained local anomalies.

To address these issues, we introduce ClipMPCAD, a CLIP-based framework for cross-domain few-shot anomaly detection (FSAD) that integrates large language model (LLM)-guided prompts and multi-attention mechanisms. Our key contributions include:

Multi-Attention Driven Feature Fusion (MADFF) module: Enhances spatial-frequency awareness and channel-level attention for precise anomaly localization.

Multi-level Semantic Decoder (M-Decoder): Combined with Professional domain Prompts (P-Prompts)—LLM-generated, domain-adaptive textual embeddings that guide hierarchical visual-text alignment.

Experiments on nine diverse datasets demonstrate the effectiveness of ClipMPCAD, achieving an average classification accuracy of 90.78% and segmentation accuracy of 98.67% on medical data, and 93.11% and 97.54% on industrial data. This work sets new benchmarks in cross-domain FSAD without additional fine-tuning.

<div align="center">
<img src="images/clipmpcad.png" width="800" alt="ClipMPCAD Architecture Diagram">
</div>

🚀 Overview
The core idea of our work is to leverage powerful large models to enhance few-shot anomaly detection. Here's a closer look at our Multi-Attention Driven Feature Fusion (MADFF) module.

<div align="center">
<img src="images/madff.png" width="800" alt="MADFF Module Diagram">
</div>

🛠️ Getting Started
This section will guide you through setting up the environment, preparing datasets, and running the code.

1. Environment Setup
Our code is built on a single NVIDIA A40 GPU. To get started, make sure you have the following dependencies installed:

python >= 3.8.5
pytorch >= 1.10.0
torchvision >= 0.11.1
numpy >= 1.19.2
scipy >= 1.5.2
kornia >= 0.6.1
pandas >= 1.1.3
opencv-python >= 4.5.4
pillow
tqdm
ftfy
regex
2. Pretrained Model
Download the pretrained CLIP model and place it in the CLIP/ckpt folder.

<p style="padding-left: 20px;">
<a href="https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt" target="_blank">ViT-L-14-336px.pt</a>
</p>

3. Prepare Datasets
The benchmark we used can be found here. We also provide pre-processed versions of the datasets for your convenience.

Download the following datasets and place them within the master directory data, then unzip them.

Liver

Brain

HIS

RESC

OCT17

ChestXray

Mvtec-AD

MPDD

BTAD

Bash

tar -xvf Liver.tar.gz
tar -xvf Brain.tar.gz
tar -xvf Histopathology_AD.tar.gz
tar -xvf Retina_RESC.tar.gz
tar -xvf Retina_OCT2017.tar.gz
tar -xvf Chest.tar.gz
tar -xvf Mvtec.tar.gz
tar -xvf MPDD.tar.gz
tar -xvf BTAD.tar.gz
4. File Structure
After the preparation work, your project structure should look like this:

code
├─ ckpt
│  ├─ few-shot
│  └─ zero-shot
├─ CLIP
│  ├─ bpe_simple_vocab_16e6.txt.gz
│  ├─ ckpt
│  │  └─ ViT-L-14-336px.pt
│  ├─ ...
├─ data
│  ├─ Brain_AD
│  └─ ...
├─ dataset
│  ├─ fewshot_seed
│  └─ ...
├─ loss.py
├─ prompt.py
├─ readme.md
├─ train_few.py
├─ train_zero.py
└─ utils.py
🚀 Quick Start
To test on a specific dataset with a few-shot number, use the test_few.py script.

Bash

python test_few.py --obj $target-object --shot $few-shot-number
For example, to test on the Brain MRI dataset with k=4, simply run:

Bash

python test_few.py --obj Brain --shot 4
To train the model, use the train_few.py script.

Bash

python train_few.py --obj $target-object --shot $few-shot-number
🖼️ Visualization
Here are some visual results from our model, showcasing its ability to accurately detect and segment anomalies.

<div align="center">
<img src="images/visual.png" width="800" alt="Visualization of Results">
</div>

🤝 Acknowledgement
We are grateful to the authors of the following projects, as we have borrowed some code from their work.

OpenCLIP

April-GAN

📧 Contact
If you have any questions, feel free to contact us.
