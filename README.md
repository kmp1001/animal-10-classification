# LiteIResNet Animal-10 Classification

A course-project implementation of a lightweight ResNet-based model for 10-class animal image classification, achieving **91%** final accuracy.  
tips:Please refer to the **“Experiment Report”** (written in Chinese, as the course is taught in Chinese) and the **presentation slides**.

---

## 🎯 Project Overview and Inspiration

The goal of this project is to design, implement and evaluate a lightweight deep-learning model for classifying 10 animal species on the Animal-10 dataset. We compare a MobileNetV3 baseline against multiple improved variants through systematic ablation studies, and arrive at a final model that achieves **91%** classification accuracy.

- **MobileNetV3 Backbone**  
  We chose MobileNetV3 for its inverted-residual blocks and lightweight activation (Hardswish), which achieve high accuracy with minimal computation—ideal for resource-constrained scenarios.

- **Efficient Channel Attention (ECAAttention)**  
  Inspired by ECA-Net, we replace the standard SE module with a 1D-convolutional channel attention that captures local cross-channel interactions with negligible extra cost, boosting the network’s ability to focus on salient features.

- **Multi-Scale Feature Fusion**  
  Drawing from feature-pyramid concepts, we aggregate intermediate representations via global average pooling to enrich high-level features with spatial details, aiming to improve recognition of animals at varied scales.

- **ArcMarginProduct Classifier**  
  Borrowed from metric-learning approaches (e.g. ArcFace), this classification head introduces an angular margin to the logits, encouraging a larger inter-class angular separation and thus more discriminative embeddings.

- **Ablation-Driven Simplification**  
  We empirically evaluate which modules genuinely contribute to performance on Animal-10. Counterintuitively, removing certain complex components often improved stability and accuracy, guiding our final streamlined design.

<p align="center">
  <img width="800" alt="model structure" 
       src="https://github.com/user-attachments/assets/3a82c9d9-befa-4d01-8c60-7d17c07193d2" />
</p>

---

## 📊 Dataset

**Animal-10**  
- **Classes (10):** dog, horse, elephant, butterfly, chicken, cat, cow, spider, sheep, squirrel  
- **Split:**  
  - Train: 14,720 images  
  - Val:   1,840 images  
  - Test:  1,840 images  
  - Ratio: 8 : 1 : 1  

---

## ⚙️ Environment

- **Language:** Python 3.11
- **Framework:** PyTorch  
- **Key Libraries:**  
  - `torchinfo`, `tqdm`, `scikit-learn`  
  - `seaborn`, `pywavelets`  
  - (AMP mixed-precision training)

---

## 🧪 Methodology

### 1. Data Processing

- **Preprocessing:**  
  - Resize to 224×224  
  - Normalize with ImageNet statistics  
- **Augmentation:**  
  - Horizontal flip, random rotation  
  - Random affine (translate 10%, scale 90–110%, shear ±10°)  
  - Random crop, color jitter  
  - Random erasing  

### 2. Model Architecture

- **Baseline:** MobileNetV3  
- **Improved “LiteIResNet” Variants:**  
  - Replace SE with **ECAAttention**  
  - Add multi-scale feature fusion  
  - Replace classifier with **ArcMarginProduct**  
- **Ablation Modules:**  
  1. Remove ECAAttention  
  2. Remove multi-scale fusion  
  3. Replace ArcMarginProduct → Linear  
  4. Combine (1) & (2)  
  5. Combine (1) & (3)  
  6. Combine (2) & (3)  

### 3. Ablation Studies

| Experiment                  | Key Change                             |
|-----------------------------|----------------------------------------|
| Ablation 1                  | – ECAAttention                        |
| Ablation 2                  | – Multi-scale fusion                  |
| Ablation 3                  | – ArcMarginProduct → Linear           |
| Ablation 4                  | 1 & 2                                 |
| Ablation 5                  | 1 & 3                                 |
| Ablation 6                  | 2 & 3                                 |

### 4. Training & Early Stopping

- **Loss:** Cross-entropy with label smoothing  
- **Optimizer:** AdamW  
- **Scheduler:** CosineAnnealingLR  
- **Batch Size:** 64  
- **Epochs:** up to 80 (early stop patience 10)  
- **Mixed-Precision:** AMP  

---

## 📈 Results And Experimental Results Analysis

| Configuration                | Accuracy | Precision | Recall | F1-Score |
|------------------------------|---------:|----------:|-------:|---------:|
| Baseline (MobileNetV3)       |   82.66% |    82.92% |  82.66% |   82.59% |
| Ablation 1 (–ECA)            |   83.64% |    83.84% |  83.64% |   83.55% |
| Ablation 2 (–Fusion)         |   82.01% |    82.42% |  82.01% |   81.92% |
| Ablation 3 (–Arc→Linear)     |   83.64% |    83.67% |  83.64% |   83.60% |
| Ablation 4 (1 & 2)           |   83.64% |    83.83% |  83.64% |   83.57% |
| Ablation 5 (1 & 3)           |   84.13% |    84.31% |  84.13% |   84.05% |
| **Ablation 6 (2 & 3)**       | **84.51%** | **84.92%** |**84.51%**| **84.53%** |
| **Final Model (tuned)**      | **91.00%** | —         | —       | —        |

> **Best ablation:** Exp 6 (remove fusion & use linear classifier) → 84.51%  
> **Final tuning** (further hyperparam & augmentation tweaks) → **91%**
<p align="center">
<img width="800" alt="final quantitative index table" src="https://github.com/user-attachments/assets/369a6250-d137-4135-93a9-e5131ee61c50" />
</p>
<p align="center">
<img width="800" alt="final confusion matrix" src="https://github.com/user-attachments/assets/2e26f640-2a92-4885-9073-245670a30198" />
</p>

By comparing the performance across all ablation studies, we found that, although adding complex modules (such as multi-scale fusion and ArcMarginProduct) can boost accuracy in some tasks, these enhancements did not yield significant improvements on the Animal-10 dataset—and in fact sometimes introduced training instability or overfitting. In contrast, simplifying the model led to markedly better performance and a more stable training process. This indicates that, for Animal-10, a streamlined architecture is better suited to the dataset’s characteristics.

During our experiments, we observed several key factors that significantly influenced model performance:

- **Role of Dropout**:
   Incorporating Dropout during training effectively prevented overfitting, especially on this small dataset. By randomly “dropping” neurons, Dropout forces the network to learn more robust feature representations and reduces over-reliance on any single pattern. This regularization strategy yielded a clear improvement in validation accuracy.

- **Batch Size & Learning Rate Tuning**: Reducing the batch size increased the noise in gradient estimates, which helped the model escape local minima and improved generalization. Likewise, careful adjustment of the learning rate schedule proved critical for both training stability and final accuracy. In practice, we tuned the learning rate decay to achieve smoother convergence and faster training.

- **Importance of Data Augmentation**: To bolster robustness, we applied a suite of augmentations (e.g., horizontal flips, random rotations, affine transforms). These operations substantially increased data diversity, helping the model handle variations in input and reducing the risk of overfitting.

**Hyperparameter Optimization** (the “trick” that boosted accuracy from 85% → 91%)
We attribute the jump from ~85% to 91% accuracy largely to targeted hyperparameter optimizations:

- **Dropout Usage**:Dropout is a proven regularization technique that randomly disables a subset of neurons each batch, preventing co-adaptation and overfitting.

- **Smaller Batch Size & Learning Rate Scheduling**:A reduced batch size introduces beneficial gradient noise that can help the model escape shallow minima and improve generalization.  Adjusting the learning rate schedule (e.g., using cosine annealing) further stabilized training and accelerated convergence.

- **Addressing Class Imbalance & Feature Similarity**:We noticed particularly low accuracy on class 0 and other visually similar classes. Recognizing this guided us to refine our augmentation and sampling strategies.

- **Enhanced Augmentation Focus**:Emphasizing random cropping and color jitter increased the effective size and variety of our dataset, directly contributing to improved generalization on unseen samples.

---

## 🏁 Conclusion

Through systematic ablation and targeted improvements on a MobileNetV3 backbone, our LiteIResNet-based model achieves **91%** accuracy on Animal-10. We demonstrate that, for this dataset, simplifying certain modules can yield both performance gains and training stability.

---

## 📚 References

1. Wang, Qilong, et al. "ECA-Net: Efficient channel attention for deep convolutional neural networks." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.
2. Sinha, Debjyoti, and Mohamed El-Sharkawy. "Thin mobilenet: An enhanced mobilenet architecture." 2019 IEEE 10th annual ubiquitous computing, electronics & mobile communication conference (UEMCON). IEEE, 2019.
3. Koonce, Brett, and Brett Koonce. "ResNet 50." Convolutional neural networks with swift for tensorflow: image recognition and dataset categorization (2021): 63-72.
  
```

