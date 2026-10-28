# Autism_Detection
Autism Detection Prototype Using EEG Features and M-CHAT
A research prototype that combines EEG-derived features and M-CHAT behavioral questionnaire responses to explore autism spectrum disorder (ASD) classification. The notebook implements a multimodal deep-learning architecture with graph-based EEG feature processing, attention-based fusion, fold-specific synthetic data augmentation, and Leave-One-Subject-Out (LOSO) evaluation.
Important: This is an experimental research prototype, not a medical device or diagnostic tool. It must not be used to diagnose ASD or make clinical decisions. The dataset contains only 28 real children, so all performance estimates are preliminary and highly uncertain.

Overview
The project explores whether EEG statistical features and behavioral questionnaire responses can be combined to distinguish ASD and control classes.

The notebook:
- Loads EEG-derived features and M-CHAT responses for 28 real children.
- Represents eight EEG channels using three statistical features per channel.
- Builds an EEG-channel graph for each child.
- Uses a GNN and attention-based Transformer layers to learn and fuse multimodal representations.
- Generates synthetic training examples using only the training portion of each LOSO fold.
- Evaluates predictions on real, held-out children.
- Compares the deep-learning model with classical machine-learning baselines.
- Trains an ensemble of final models and saves model weights and scalers.
Model Architecture
The main model is EEGNetGNNFTTransformer.
1. EEG feature projection: Each of the 8 EEG channels has 3 features: coefficient of variation, entropy, and skewness. A linear projection maps channel features into a 32-dimensional representation.
2. Graph neural network: Two graph-convolution layers aggregate information across the eight EEG channels using an adjacency matrix derived from correlations between channel feature vectors.
3. M-CHAT feature tokenization: The 23 questionnaire answers are converted into 32-dimensional tokens.
4. Cross-attention fusion: EEG graph representations attend to M-CHAT tokens, combining information from both modalities.
5. Self-attention: Attention is applied across the fused EEG-channel representations.
6. Pooling and classification: Mean pooling and a feed-forward classifier produce a logit, which is converted to an ASD-class probability during prediction.
High-level pipeline
EEG statistical features (8 channels × 3 features)
                  │
          EEG feature projection
                  │
       Graph convolution × 2  ◄── EEG channel adjacency graph
                  │
                  ├───────────────┐
                  │               │
                  │        M-CHAT answers (23)
                  │               │
                  │        Feature tokenizer
                  │               │
                  └── Cross-attention fusion
                                  │
                              Self-attention
                                  │
                              Mean pooling
                                  │
                           Binary classifier
                                  │
                         ASD-class probability
Dataset and Input Features
The notebook expects the dataset archive A_Multi-Modal_Dataset_of_EEG_Signals_and_M-CHAT_As.zip (or an already extracted dataset folder).

It searches for the Extracted Features directory and reads the following feature files:
- Coefficient Of Variation.csv
- Entropy.csv
- Skewness.csv
It also loads:
- ASD-M-CHAT.csv
- CTRL-M-CHAT.csv
The expected folder structure includes:
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
The notebook matches participants across EEG features and questionnaire data. It uses 28 real children in the current pipeline.
Input	Shape per child	Description
EEG features	8 × 3	Eight channels; coefficient of variation, entropy, and skewness
M-CHAT responses	23	Responses to 23 questionnaire items
Target	1	Class label: 1 = ASD, 0 = Control


The archive's exact directory layout may vary; the notebook's dataset-discovery logic should be checked if the files are organized differently.
Synthetic Data Augmentation
Synthetic examples are generated separately inside each training fold:
- Two real training children from the same class are selected as parents.
- Their EEG features are blended and small random noise is added.
- Each synthetic M-CHAT answer is copied from one of the two parents, preserving binary responses.
- The synthetic example keeps the parents' class label.
- The held-out child is not used to create synthetic examples.
The notebook creates five synthetic examples per real training child (N_SYNTH_PER_REAL = 5). The EEG scaler is fitted using only the real training children in each fold.

Synthetic examples are derived from a very small dataset and are not equivalent to collecting additional real participants. Their usefulness must be established through rigorous validation.
Evaluation Strategy
The notebook uses Leave-One-Subject-Out (LOSO) cross-validation:
1. Hold out one real child for testing.
2. Use the other 27 real children for training.
3. Fit preprocessing on the real training subset.
4. Generate synthetic data using only that fold's training children.
5. Train the model and predict the held-out real child.
6. Repeat until each real child has been held out.
The deep model is repeated across 10 seeds (SEEDS = list(range(10)) and the notebook reports metrics including:
- Accuracy
- ROC-AUC
- Sensitivity (recall for the ASD class)
- Specificity
- Mean and standard deviation across seeds
The notebook also compares classical models—including Logistic Regression, RBF SVM, k-Nearest Neighbours, Random Forest, Extra Trees, Gradient Boosting, and XGBoost when installed—under the same LOSO framework, with real-only and real-plus-synthetic training conditions.

Interpretation caution: With only 28 children, one child's prediction changes overall accuracy by about 3.6 percentage points. Results should be treated as preliminary internal validation, not evidence of clinical readiness. The notebook itself notes that a permutation test has not yet been confirmed. Repeated seeds do not increase the number of independent participants.
Requirements
The notebook uses Python and the following main packages:
- NumPy
- pandas
- Matplotlib
- seaborn
- scikit-learn
- PyTorch
- joblib
- XGBoost (for the XGBoost baseline; optional if unavailable)
The first setup cell installs seaborn, matplotlib, scikit-learn, and joblib; the comparison section attempts to install xgboost.
How to Run
Option 1: Google Colab
1. Upload Autism_synthetic_pipeline.ipynb to Google Colab.
2. Upload the dataset ZIP when prompted, or place it in the notebook's working directory.
3. Run the notebook cells in order, from environment setup through final model training.
4. Review the LOSO evaluation and baseline comparison outputs.
5. Check the saved_models/ directory for saved model weights and scalers.
Option 2: Local Jupyter environment
1. Install Python and Jupyter Notebook/Lab.
2. Install the required dependencies.
3. Place the dataset ZIP in the working directory, or extract it so the notebook can locate Extracted Features.
4. Open the notebook and run the cells in order.
The notebook may require small adjustments for local execution because its dataset-upload fallback uses Google Colab's files.upload() when running in Colab.
Outputs
The notebook is designed to produce:
- LOSO evaluation metrics for the deep-learning model.
- A comparison table and chart for classical baselines.
- model_comparison.csv containing baseline comparison results.
- A saved_models/ directory containing five model checkpoints and five EEG scalers:
  - asd_model_seed0.pth through asd_model_seed4.pth
  - eeg_scaler_seed0.joblib through eeg_scaler_seed4.joblib
The final prediction function, predict_new_child(eeg_features_8x3, mchat_23), returns a mean score and the standard deviation across five models. This score is an experimental model output, not a calibrated clinical risk estimate. The notebook's demonstration uses a child already included in training and should not be interpreted as an independent test result.
Reproducibility Notes
- A fixed global seed (SEED = 42) is set during setup.
- Training and evaluation use multiple seeds.
- The model selects CUDA when available and otherwise uses CPU.
- Results can vary with package versions, hardware, and nondeterministic GPU operations.
- Keep the original dataset's documentation and usage terms with the data.
Limitations and Future Work
- Very small sample size: only 28 real children are included in the current pipeline.
- Synthetic data may not reflect the variability of real-world participants.
- Results need independent external validation on a substantially larger and more diverse cohort.
- Statistical uncertainty and potential class imbalance should be reported clearly.
- A permutation test and additional robustness checks are needed before making stronger performance claims.
- Clinical use would require appropriate clinical validation, regulatory review, privacy safeguards, and expert oversight.
Potential next steps include testing on an independent cohort, reporting confidence intervals, running permutation tests, checking calibration, documenting the dataset source and consent/privacy conditions, and comparing against simpler baselines without synthetic augmentation.
Repository Structure
.
├── Autism_synthetic_pipeline.ipynb
├── README.md
├── model_comparison.csv        # generated after running the notebook
└── saved_models/               # generated after final model training
    ├── asd_model_seed0.pth
    ├── ...
    └── eeg_scaler_seed4.joblib
Generated outputs will appear only after the relevant notebook cells have been run.
Disclaimer
This repository is intended for educational and research exploration only. It does not provide medical advice, does not diagnose autism, and should not be used to screen or make decisions about any individual. Any real-world application would require validation by qualified clinical researchers and professionals.
License
Add a license before distributing or reusing this repository. Ensure you have permission to share the dataset and comply with its original license, consent, and privacy requirements.
