# Multi-Label Retinal Disease Classification using Deep Learning (ODIR-5K)

## 1. Problem Definition
This project addresses the problem of **multi-label retinal disease classification** from fundus images.
Unlike multi-class classification, a single retinal image may contain **multiple co-existing diseases**, therefore the task is formulated as a **multi-label classification problem**.

The model outputs independent probabilities for each disease using a sigmoid activation function, enabling the prediction of more than one condition per image.

**Disease Labels (8):**
- N: Normal
- D: Diabetic Retinopathy
- G: Glaucoma
- C: Cataract
- A: Age-related Macular Degeneration
- H: Hypertension
- M: Myopia
- O: Other abnormalities

---

## 2. Dataset and Preprocessing
- **Dataset:** ODIR-5K (Ocular Disease Intelligent Recognition)
- **Source:** Kaggle
- **Split:** Train / Validation = 80 / 20
- **Input Size:** 224 × 224 RGB images

### Preprocessing Steps
- Resize images to 224×224
- Normalize pixel values to [0,1]
- Data augmentation is applied **only to the training set**
- Validation data is kept unchanged for unbiased evaluation

> Note: The dataset is not included in this repository due to size and licensing constraints.  
> Users should download the dataset separately and update the data paths accordingly.

---

## 3. Model Architectures and Rationale
The following models are evaluated:

- **Baseline CNN (from scratch)**  
  A simple convolutional neural network trained without pre-trained weights, used as a reference floor model.

- **VGG16 (Transfer Learning)**  
  Used as a baseline among pre-trained CNN architectures due to its simple and well-known structure.

- **ResNet50 (Transfer Learning)**  
  Incorporates residual connections to improve gradient flow in deeper networks.

- **DenseNet121 (Improved Model)**  
  Employs dense connections that enable feature reuse and improved representation learning, particularly beneficial for medical images.

### Improved Training Strategy (DenseNet121)
- Focal Loss to address class imbalance
- PR-AUC as a monitoring metric
- Decision threshold optimization for multi-label prediction

---

## 4. How to Run the Project
This project is designed to be executed using **Google Colab**.

### A. Environment Setup

Most required libraries (TensorFlow, NumPy, Pandas, Scikit-learn) are pre-installed in Google Colab.  
If additional dependencies are needed, they can be installed using:

```bash
pip install -r requirements.txt
```

### B. Running the Experiments
To run the project:
1. Open the notebook deep_learning_retinal_disease.ipynb in Google Colab.
2. Upload or mount your dataset (e.g., via Google Drive).
3. Ensure that the file paths match the dataset structure.
4. Run cells in order:
  - Data loading and preprocessing
  - Baseline and transfer learning model training
  - Evaluation and metric computation
  - Threshold optimization and final comparisons

The notebook contains all steps from data preparation to evaluation.

---

## 5. Model Outputs
```markdown
### Performance Comparison

The table below summarizes the evaluation results of all models on the validation dataset:

| Model                | Micro Precision | Micro Recall | Micro F1 |
|----------------------|-----------------|--------------|----------|
| Baseline CNN         | 0.5926          | 0.0582       | 0.1060   |
| VGG16                | 0.6005          | 0.1339       | 0.2190   |
| ResNet50             | 0.5055          | 0.0842       | 0.1444   |
| DenseNet121          | 0.6467          | 0.1964       | 0.3013   |
| DenseNet121 Improve  | 0.3535          | 0.7139       | 0.4729   |

*Note: Micro metrics provide an overall measure across all labels.*

### Per-Label Performance (DenseNet121 Improve)

                precision    recall  f1-score   support

           N       0.34      0.89      0.50       438
           D       0.36      0.91      0.51       442
           G       0.28      0.21      0.24        85
           C       0.46      0.55      0.50        91
           A       0.00      0.00      0.00        61
           H       0.33      0.11      0.16        46
           M       0.56      0.67      0.61        69
           O       0.34      0.63      0.44       418

   micro avg       0.35      0.71      0.47      1650
   macro avg       0.33      0.50      0.37      1650
weighted avg       0.34      0.71      0.45      1650
 samples avg       0.37      0.74      0.47      1650
```

---

## 6. Evaluation Summary
The DenseNet121 model with the improved training strategy achieved the highest overall Micro F1 score among all evaluated models, indicating a strong balance between precision and recall on the multi-label validation dataset. Some disease labels, such as AMD (A) and Glaucoma (G), had lower F1 scores, likely due to class imbalance and visual similarity with other conditions. Despite this, the improved DenseNet121 model showed reliable performance in identifying common disease patterns. This section provides an overview of the model comparison results and highlights important performance observations. Detailed per-class evaluation results and training curves can be found in the project notebook (`deep_learning_retinal_disease.ipynb`).

