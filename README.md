# 🧠 AN2DL Challenges 2025/2026

## 🎓 Overview
This repository contains the project code, models, and reports developed for the **Artificial Neural Network and Deep Learning (AN2DL)** course during the 2025/2026 academic year as part of the **MSc in Computer Science & Engineering at Politecnico di Milano**. 

The work here represents the collective effort of team **Batch Size Matters** (Matteo Rossi, Riccardo Piantoni, Jacopo Sacramone, and Angelo Prete) across two model competitions.

---

## 🏆 Competition Results

Across both challenges, our team competed against 193 teams. The final evaluations were based on the Weighted/Macro F1 Score. 

| Challenge | Leaderboard | Rank | F1 Score | Total Teams |
| :--- | :--- | :--- | :--- | :--- |
| **Challenge 1** (Time Series) | Public | **22** | 0.93785 | 193 |
| **Challenge 1** (Time Series) | Private | **33** | 0.95809 | 193 |
| **Challenge 2** (Computer Vision) | Unique | **10** | 0.44890 | 193 |

---

## 🏃‍♂️ Challenge 1: Patient Pain Level Classification

### 📊 The Task
The first challenge focused on the automatic classification of patients' pain levels from sensor data collected during physical movements. We framed this as a **Multivariate Time Series Classification** problem. 
Given a temporal sequence of feature vectors representing 31 continuous joint measurements and 4 aggregated pain indicators over 160 time steps, the objective was to classify the patient into one of three discrete categories: `no_pain`, `low_pain`, or `high_pain`.

### 🧩 Challenges
* **Lack of Context**: Limited information on the data acquisition process, sensor placement, and spatial symmetries made it difficult to interpret numerical values and group sensors by anatomy.
* **Dimensionality & Data Scarcity**: High-dimensional patient data paired with a relatively small training set (661 patients) hindered complex pattern learning.
* **Noisy Data**: The presence of severe sensor glitches and pronounced spikes on specific joints required careful anomaly detection and linear interpolation.

### 🚀 Our Approach
We developed a custom **Two-Stage Deep Architecture** to capture both local window patterns and global sequence context:
1. **First Stage (Window-Level Representation)**: Raw joint signals are processed through a 1D Convolutional layer to extract higher-level features. These features are fed into a Bidirectional GRU (BiGRU) equipped with an Attention mechanism and an MLP head, generating window-level class logits.
2. **Second Stage (Sequence Meta-Model)**: The sequences of Out-Of-Fold (OOF) logits from the first stage are fed into another BiGRU with Attention. This meta-model learns to optimally weight and combine temporal information across the entire sequence instead of relying on a naive average.

We also employed **Label Smoothing** and a **Class-Weighted Cross-Entropy Loss** to encourage calibrated predictions and tackle class imbalance effectively.

---

## 🔬 Challenge 2: Breast Cancer Subtype Classification

### 📊 The Task
The second challenge shifted the domain to Computer Vision. The task involved analyzing microscopic tissue morphology images to predict four distinct breast cancer molecular subtypes: **Luminal A**, **Luminal B**, **HER2(+)**, and **Triple negative**.

### 🧩 Challenges
* **Image Variance**: The dataset consisted of 1,272 tissue images of highly variable sizes.
* **Irrelevant Backgrounds**: Large portions of the images contained non-pathological background, requiring reliance on binary masks to locate diseased regions.
* **Low Resolution**: Compared to standard whole-slide images (WSIs) that typically offer 20k-30k pixel resolution, this dataset presented low-resolution constraints.
* **Computational Bottlenecks**: Processing massive images sequentially made standard CNN/ViT fine-tuning unfeasibly slow for hyperparameter search.

### 🚀 Our Approach
To overcome these limitations, we engineered an **Attention-Based Multiple Instance Learning (MIL)** pipeline combined with offline feature extraction:
1. **Patch Extraction**: Guided by binary masks, we extracted fixed-size square patches around masked regions, filtering out background-heavy patches using a coverage score threshold.
2. **Feature Precomputation (Offline)**: 
   * **Images**: Patches were embedded using **UNI2-h**, a state-of-the-art Vision Transformer pre-trained explicitly on histopathology slides.
   * **Masks**: The binary masks corresponding to the patches were embedded using a single-channel **ConvNeXt-Tiny** network.
3. **Two-Tower MIL Architecture**: The precomputed image and mask features were fused and processed by an attention-based MIL module. Instead of naive averaging, the MIL layer learned to score and aggregate the most critical patches within an image "bag" to produce the final patient-level classification.

Extensive data augmentations at the patch level (flips, rotations, AugMix) combined with Optuna hyperparameter tuning heavily boosted our final model performance.
