# Fine-Tuning the Parakeet RNNT 0.6B Model on the Urdu Common Voice Dataset

This repository showcases the fine-tuning process of NVIDIA's **Parakeet RNNT 0.6B** model on the **Urdu dataset** from Mozilla's **Common Voice**. The fine-tuning was conducted to improve the model's Automatic Speech Recognition (ASR) capabilities for Urdu, producing promising results even with limited training time.

---

## Model Overview

### Parakeet RNNT
The **Parakeet RNNT** (Recurrent Neural Network Transducer) model is an XL version of the FastConformer Transducer. It boasts **600 million parameters**, enabling state-of-the-art ASR for speech-to-text tasks. Developed by NVIDIA and suno.ai, it specializes in transcribing speech in **lowercase English**.

You can find the base model on 🤗 [Hugging Face](https://huggingface.co/nvidia/parakeet-rnnt-0.6b).

### FastConformer
The **FastConformer** architecture, introduced by NVIDIA, builds on Google's **Conformer** model, combining:
- **Self-attention mechanisms** for capturing long-range dependencies.
- **Convolutional layers** for local and sequential information.

For details on FastConformer, refer to the [research paper](https://arxiv.org/pdf/2305.05084).

---

## Data

The fine-tuning process utilized the **Urdu dataset** from Mozilla's Common Voice, accessed via the 🤗 [Hugging Face Datasets Library](https://huggingface.co/datasets/mozilla-foundation/common_voice_12_0). This dataset provides a diverse range of Urdu speech samples, ensuring robust training.

---

## Training Resources

### Hardware
- **Google Colab Pro**: Fine-tuning was performed on an **NVIDIA A100 GPU** over approximately **5 hours**.  
- The GPU was utilized at only one-quarter of its capacity, so the time can be cut down more.

### Library Installations
To avoid version conflicts, the required Python packages were installed using the provided `pip` commands. Ensure your **NVCC driver version is 16.0 or higher** when running the notebook locally.

---

## Results

The fine-tuned model achieved a **Word Error Rate (WER)** of **25.513%**, which might seem high at first glance. However, considering that WER for Whisper is **23%** and the fact that transcriptions show remarkable accuracy in many cases:

- **Reference**: کچھ بھی ہو سکتا ہے۔  
  **Predicted**: کچھ بھی ہو سکتا ہے۔  

---

- **Reference**: اورکوئی جمہوریت کو کوس رہا ہے۔  
  **Predicted**: اور کوئ جمہوریت کو  کو س رہا ہے۔  

We can say that results are impressive given the limited fine-tuning time and highlight the potential for further refinement.

---
You can find the fine tuned model on 🤗 [Hugging Face](https://huggingface.co/hash2004/parakeet-fine-tuned-urdu). 