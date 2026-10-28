# Autism Detection Using EEG and M-CHAT

## 📌 Project Overview

This project explores autism spectrum disorder (ASD) classification using EEG-derived statistical features and M-CHAT behavioral questionnaire responses. It combines Graph Neural Networks (GNNs), attention-based Transformer layers, and synthetic data augmentation to investigate multimodal learning.

The project uses Leave-One-Subject-Out (LOSO) cross-validation to evaluate model predictions on held-out participants.

**Note:** This is an experimental research prototype, not a medical diagnostic tool.

## 🎯 Objectives

- Analyze EEG features for autism-related classification.
- Combine EEG signals' statistical features with M-CHAT questionnaire responses.
- Develop a multimodal deep-learning architecture.
- Explore synthetic data augmentation for model training.
- Evaluate performance using subject-wise cross-validation.
- Compare deep learning with traditional machine-learning models.

## 🧠 Model Architecture

The main model, `EEGNetGNNFTTransformer`, combines EEG and behavioral data.

1. **EEG Feature Extraction:** Uses coefficient of variation, entropy, and skewness from eight EEG channels.
2. **Graph Neural Network (GNN):** Learns relationships between EEG channels using an adjacency graph.
3. **M-CHAT Feature Processing:** Converts 23 questionnaire responses into feature representations.
4. **Cross-Attention Fusion:** Combines EEG and behavioral information.
5. **Self-Attention:** Learns relationships among EEG channel representations.
6. **Classification Layer:** Produces a prediction score for the ASD class.

### Architecture Diagram

```text
EEG Features (8 Channels × 3 Features)
                  |
          Feature Projection
                  |
          Graph Neural Network
                  |
           EEG Representation
                  |
                  +------------------+
                  |                  |
                  |            M-CHAT Responses
                  |                  |
                  |            Feature Tokenizer
                  |                  |
                  +---- Cross-Attention
                           Fusion
                             |
                       Self-Attention
                             |
                        Mean Pooling
                             |
                     Classification Layer
                             |
                       Prediction Score
```

## 📊 Dataset

The notebook uses a multimodal dataset containing EEG-derived features and M-CHAT questionnaire responses.

### EEG Features

- Coefficient of Variation
- Entropy
- Skewness

### Behavioral Features

- 23 M-CHAT questionnaire responses

### Dataset Size

- 28 real children in the current pipeline.
- Eight EEG channels.
- Three statistical features per EEG channel.
- 23 M-CHAT responses per participant.

### Expected Dataset Structure

```text
dataset/
├── Extracted Features/
│   ├── Coefficient Of Variation.csv
│   ├── Entropy.csv
│   └── Skewness.csv
└── Raw Data/
    ├── Raw ASD Data/
    │   └── Behavioral Data/
    │       └── ASD-M-CHAT.csv
    └── Raw Control Data/
        └── Behavioral Data/
            └── CTRL-M-CHAT.csv
```

The notebook expects the dataset archive `A_Multi-Modal_Dataset_of_EEG_Signals_and_M-CHAT_As.zip`, or an extracted dataset folder containing the required files.

## 🔬 Synthetic Data Augmentation

The project generates synthetic training samples from real participants to explore the effect of data augmentation.

- Two real training participants from the same class are selected.
- Their EEG features are blended with small random noise.
- M-CHAT responses are sampled from the corresponding parent responses.
- Synthetic samples retain their parents' class label.
- Synthetic data is generated separately within each training fold.

The notebook creates five synthetic samples per real training participant.

Synthetic samples do not replace real participants, and their effectiveness requires further validation.

## 🧪 Model Evaluation

The project uses **Leave-One-Subject-Out (LOSO) Cross-Validation**.

### Evaluation Process

1. One real participant is held out for testing.
2. The remaining participants are used for training.
3. Preprocessing is fitted using the training participants.
4. Synthetic samples are generated from training data only.
5. The model predicts the held-out participant.
6. The process repeats until every real participant has been held out.

The deep-learning model is evaluated across multiple random seeds.

### Evaluation Metrics

- Accuracy
- ROC-AUC
- Sensitivity
- Specificity
- Mean and standard deviation across seeds

### Baseline Models

The notebook also compares traditional machine-learning approaches:

- Logistic Regression
- Support Vector Machine (RBF kernel)
- K-Nearest Neighbours
- Random Forest
- Extra Trees
- Gradient Boosting
- XGBoost (when available)

## 🛠️ Technologies Used

- Python
- PyTorch
- NumPy
- pandas
- scikit-learn
- Matplotlib
- seaborn
- joblib
- XGBoost (optional)

## 🚀 How to Run the Project

### Option 1: Google Colab

1. Upload `Autism_synthetic_pipeline.ipynb` to Google Colab.
2. Upload the required dataset ZIP when prompted, or place it in the working directory.
3. Execute the notebook cells in order.
4. Review the model evaluation and baseline comparison results.
5. Check the generated output files.

### Option 2: Jupyter Notebook

1. Install Python and Jupyter Notebook.
2. Install the required Python packages.
3. Place the dataset ZIP or extracted dataset in the expected location.
4. Open `Autism_synthetic_pipeline.ipynb`.
5. Run the cells sequentially.

Some dataset-upload code may require adjustment when running outside Google Colab.

## 📁 Project Structure

```text
Autism-Detection/
├── Autism_synthetic_pipeline.ipynb
├── README.md
├── model_comparison.csv
└── saved_models/
    ├── asd_model_seed0.pth
    ├── asd_model_seed1.pth
    ├── asd_model_seed2.pth
    ├── asd_model_seed3.pth
    ├── asd_model_seed4.pth
    └── eeg_scaler_seed0.joblib
```

The CSV file and saved model artifacts are generated after running the relevant notebook cells. The scaler directory also contains the remaining saved scalers.

## 💾 Model Outputs

The notebook generates:

- LOSO evaluation results.
- Baseline model comparison results.
- `model_comparison.csv`.
- Trained model checkpoints.
- EEG feature scalers.
- Prediction scores and model-ensemble statistics.

The function `predict_new_child(eeg_features_8x3, mchat_23)` produces an experimental score and variation across five saved models.

This output is not a clinically validated probability or diagnosis.

## ⚠️ Limitations

- The current pipeline contains only 28 real participants.
- Synthetic data may not represent real-world participant variability.
- Performance estimates are uncertain because of the small sample size.
- Independent external validation is required.
- Further statistical testing and robustness checks are needed.
- Clinical use would require rigorous validation and appropriate expert oversight.

## 🔮 Future Improvements

- Evaluate the model on a larger, independent dataset.
- Add confidence intervals and statistical significance testing.
- Perform permutation tests and additional robustness checks.
- Evaluate probability calibration.
- Compare performance with and without synthetic augmentation.
- Improve reproducibility and document dataset permissions.

## ⚕️ Disclaimer

This project is intended for educational and research purposes only. It is not intended to diagnose autism, provide medical advice, or replace professional clinical assessment. Do not use its predictions to make medical or screening decisions about individuals.

## 📄 License

Add an appropriate license before distributing or reusing this project. Ensure that the dataset's original license, consent requirements, privacy restrictions, and usage conditions are respected.
