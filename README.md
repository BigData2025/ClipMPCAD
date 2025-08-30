<div align="center"> <h1> 📌 ClipMPCAD </h1> <h3>Few-Shot Anomaly Detection with LLM-Guided Prompts and Multi-Attention Fusion</h3> </div>
<div align="center"> <img src="images/clipmpcad.png" width="60%"> </div>
⭐ Abstract

Anomaly detection in medical and industrial images often faces challenges such as limited data availability and strict privacy constraints, which hinder large-scale model training. Although pre-trained large vision-language models (VLMs) offer promise in few-shot anomaly detection (FSAD), their effectiveness is limited by weak cross-modal alignment, insufficient domain-specific supervision, and poor sensitivity to fine-grained local anomalies.

To address these issues, we propose ClipMPCAD, a CLIP-based framework for cross-domain FSAD that integrates large language model (LLM)-guided prompts and multi attention mechanisms. Specifically, we introduce:

Multi-Attention Driven Feature Fusion (MADFF) module that enhances spatial-frequency awareness and channel-level attention for precise localization.

Multi-level Semantic Decoder (M-Decoder) combined with Professional domain Prompts (P-Prompts) — LLM-generated, domain-adaptive textual embeddings that guide hierarchical visual-text alignment.

Experiments on nine diverse datasets demonstrate the effectiveness of ClipMPCAD, achieving average classification & segmentation accuracies of 90.78% and 98.67% on medical data and 93.11% and 97.54% on industrial data, setting new benchmarks in cross-domain FSAD without additional fine-tuning.

Keywords: Few-Shot, Large Vision-Language Model, Anomaly Classification and Segmentation

<div align="center"> <img src="images/madff.png" width="80%"> </div>
🚀 Get Started
⚙️ Environment

python >= 3.8.5

pytorch >= 1.10.0

torchvision >= 0.11.1

numpy >= 1.19.2

scipy >= 1.5.2

kornia >= 0.6.1

pandas >= 1.1.3

opencv-python >= 4.5.4

pillow, tqdm, ftfy, regex

💻 Device

Single NVIDIA A40 GPU

📦 Pretrained model

CLIP ViT-L-14-336px

Put the file under CLIP/ckpt folder.

🏥 Medical and Industrial Anomaly Detection Benchmark

(optional) Follow BMAD
 to apply for permission to download the dataset.

We provide the pre-processed benchmark:

Liver: ----------------------------------------------------------------------------------

Brain: ----------------------------------------------------------------------------------

HIS: ----------------------------------------------------------------------------------

RESC: ----------------------------------------------------------------------------------

OCT17: ----------------------------------------------------------------------------------

ChestXray: ----------------------------------------------------------------------------------

Mvtec-AD: ----------------------------------------------------------------------------------

MPDD: ----------------------------------------------------------------------------------

BTAD: ----------------------------------------------------------------------------------

Place them under data folder and unzip:

tar -xvf Liver.tar.gz
tar -xvf Brain.tar.gz
tar -xvf Histopathology_AD.tar.gz
tar -xvf Retina_RESC.tar.gz
tar -xvf Retina_OCT2017.tar.gz
tar -xvf Chest.tar.gz
tar -xvf Mvtec.tar.gz
tar -xvf MPDD.tar.gz
tar -xvf BTAD.tar.gz

📂 File Structure
code
├─ ckpt
│  ├─ few-shot
│  └─ zero-shot
├─ CLIP
│  ├─ bpe_simple_vocab_16e6.txt.gz
│  ├─ ckpt
│  │  └─ ViT-L-14-336px.pt
│  ├─ clip.py
│  ├─ model.py
│  ├─ models.py
│  ├─ model_configs
│  │  └─ ViT-L-14-336.json
│  ├─ modified_resnet.py
│  ├─ openai.py
│  ├─ tokenizer.py
│  └─ transformer.py
├─ data
│  ├─ Brain_AD
│  │  ├─ valid
│  │  └─ test
│  ├─ ...
│  └─ Retina_RESC_AD
│     ├─ valid
│     └─ test
├─ dataset
│  ├─ fewshot_seed
│  │  ├─ Brain
│  │  ├─ ...
│  │  └─ Retina_RESC
│  ├─ medical_few.py
│  └─ medical_zero.py
├─ loss.py
├─ prompt.py
├─ readme.md
├─ train_few.py
├─ train_zero.py
└─ utils.py

⚡ Quick Start
python test_few.py --obj $target-object --shot $few-shot-number


👉 Example: to test on Brain MRI with k=4:

python test_few.py --obj Brain --shot 4

🏋️ Training
python train_few.py --obj $target-object --shot $few-shot-number

🖼️ Visualization
<div align="center"> <img src="images/visual.png" width="70%"> </div>
🙏 Acknowledgement

We borrow some codes from OpenCLIP
, and April-GAN
.

📬 Contact

If you have any problem with this code, please feel free to contact **** and ****.
