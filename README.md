ECG Signal Forecasting using Transformer Models
 Overview
This repository implements an ECG signal forecasting pipeline using a Transformer-based deep learning model from the DARTS time-series framework. The project focuses on learning temporal dependencies in normalized ECG signals and predicting future signal values using sequence-to-sequence modeling.
The work demonstrates how transformer architectures can be applied to biomedical time-series prediction, particularly ECG signals, with emphasis on preprocessing, temporal modeling, and quantitative evaluation.

Methodology
1. Data Loading & Preprocessing
ECG signal is loaded from a CSV file.
The signal is flattened and normalized using MinMax Scaling.
A synthetic time index is generated at 1-second intervals.
The ECG signal is converted into a TimeSeries object compatible with DARTS.

2. Covariate Generation
Datetime-based covariates are created using minute-level information.
Covariates are one-hot encoded and provided as past covariates to the model.
These covariates help the transformer learn periodic temporal patterns.

3. Train–Validation Split
The time series is split into:
Training set: all but last 1000 points
Validation set: last 1000 points
Input and output window sizes:
input_chunk_length = 100
output_chunk_length = 20

 Model Architecture
The forecasting model is a TransformerModel with the following configuration:
Encoder–Decoder Transformer architecture
Multi-head self-attention
Gaussian likelihood for probabilistic forecasting
Key parameters:
d_model = 64
nhead = 4
num_encoder_layers = 2
num_decoder_layers = 2
dropout = 0.3
batch_size = 32
epochs = 10
Loss function: Mean Squared Error (via Gaussian likelihood)
The model contains approximately 372K trainable parameters.

 Training & Forecasting
The transformer is trained on the ECG training series with past covariates.
After training, the model autoregressively predicts the validation segment.
Forecasts are compared against ground truth ECG signals.

 Evaluation
Metric Used
Root Mean Square Error (RMSE)
Result
Transformer RMSE: 0.3842
This value reflects how closely the predicted ECG signal follows the actual normalized signal.

Visualization
Two plots are generated:
Full-length plot: shows complete signal and forecast (visually dense)
Zoomed validation plot: focuses on recent ECG trends to better understand prediction direction and behavior
The zoomed visualization improves interpretability by reducing visual clutter while preserving trend information.

Notes & Limitations
Training was limited to 10 epochs due to computational constraints (CPU-only execution).
Increasing epochs may improve performance but significantly increases training time.
This implementation focuses on signal forecasting, not disease classification.
Dataset details (sampling rate, patient labels) are not incorporated in this version.

Future Work
Extend to classification tasks (e.g., arrhythmia detection)
Experiment with longer input windows and deeper transformers
Compare with LSTM/GRU-based models
Incorporate additional ECG features (frequency-domain, wave morphology)
Train on GPU for larger-scale experiments

Tech Stack
Python
DARTS
PyTorch Lightning
Scikit-learn
NumPy, Pandas, Matplotlib

Application Domain
Biomedical Signal Processing
Healthcare AI
Time-Series Forecasting
