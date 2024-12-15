# Speech-Emotion-Recognition-using-Wav2Vec2

You may refer to the [report](slides.pdf) for more details on the project results.

## Project Overview
This project implements a Speech Emotion Recognition (SER) system using the Wav2Vec2 pre-trained model. The system analyzes speech audio and classifies it into four emotional categories: Neutral, Happy, Sad, and Angry. This implementation is based on the paper "3-D Convolutional Recurrent Neural Networks With Attention Model for Speech Emotion Recognition" but with significant modifications to use modern transformer-based architectures.

## Problem Statement
Speech Emotion Recognition is a challenging task in audio processing that requires understanding subtle variations in speech patterns, tone, and acoustic features. Traditional approaches often struggle with:
- Capturing temporal dependencies in speech
- Handling variable-length inputs
- Generalizing across different speakers
- Dealing with limited labeled data

## Solution Architecture
Our solution leverages the power of pre-trained models and modern deep learning techniques:

### Model Architecture
- **Feature Extractor**: Facebook's Wav2Vec2 pre-trained model
  - Pre-trained on 960 hours of LibriSpeech
  - Provides robust speech representations
- **Custom Classifier Head**:
  - Linear layers with batch normalization
  - Dropout for regularization
  - ReLU activation functions

### Dataset: IEMOCAP
The Interactive Emotional Dyadic Motion Capture (IEMOCAP) database is used for training and evaluation:
- **Total Sessions**: 5 sessions with 2 speakers each (1 male, 1 female)
- **Split**:
  - Training: Sessions 2-5
  - Validation: Session 1 (Male)
  - Test: Session 1 (Female)
- **Class Distribution**:
  ```
  Angry (A): 874 samples
  Happy (H): 1358 samples
  Neutral (N): 1324 samples
  Sad (S): 890 samples
  ```

### Training Configuration
- **Batch Size**: 128 (T4-15GB GPU) / 256 (A100-40GB GPU) / 16 (RTX 3090 Ti)
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: 1e-4 with scheduling
- **Loss Function**: Cross Entropy
- **Training Time**: ~30-60 minutes depending on GPU

### Data Augmentation Techniques
To improve model robustness, we implement several augmentation strategies:
1. Time Stretching
2. Pitch Shifting
3. Background Noise Injection
4. Time Masking

## Performance Metrics
Our model achieves significant improvements over the baseline:

| Model Version | Test-UAR | GPU | Time Taken |
|--------------|----------|-----|------------|
| Baseline     | 54.53%   | T4  | < 1 hour   |
| Our Model    | 69.02%   | RTX 3090 Ti  | ~45 mins   |

## Implementation Details

### Key Components
1. **Data Processing**
   - Audio loading and preprocessing
   - Feature extraction using Wav2Vec2
   - Data augmentation pipeline

2. **Model Training**
   - Custom training loop
   - Learning rate scheduling
   - Early stopping
   - Gradient clipping

3. **Evaluation**
   - Confusion matrix analysis
   - Per-class metrics
   - Cross-validation results

## Future Improvements
1. **Model Architecture**
   - Experiment with different attention mechanisms
   - Implement multi-task learning
   - Try other pre-trained models (HuBERT, WavLM)

2. **Data Processing**
   - Additional augmentation techniques
   - Cross-lingual adaptation
   - Speaker normalization

3. **Training Strategy**
   - Curriculum learning
   - Adversarial training
   - Semi-supervised learning with unlabeled data

## Getting Started

### Prerequisites
- Python 3
- pip
- CUDA capable GPU i.e. T4 or A100 (Google Colab) or any other GPU (Local Machine)

Once you have the prerequisites, you can install the required packages by running:

```bash
pip install -r requirements.txt
```




