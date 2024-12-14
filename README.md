# Meal Nutrition Analysis: Predicting Lunch Calories

## Overview
This project focuses on predicting **lunch calorie intake** using a multimodal dataset comprising:
- **Image Data**: Visual representations of meals before breakfast and lunch.
- **Time-Series Data**: Continuous Glucose Monitoring (CGM) readings.
- **Tabular Data**: Static demographic, clinical, and microbiome (Viome) features.

By leveraging these diverse data modalities, the project aims to create a robust machine learning pipeline capable of accurate calorie estimation.

---

## Features
### Multimodal Data Processing
- **Images**: 
  - RGB images of meals, resized and normalized for model input.
  - Channels for breakfast and lunch concatenated into a single tensor.
- **Time-Series Data**:
  - Glucose readings within a 2-hour window before and after meals.
  - Padded or truncated to a fixed sequence length of 48 timesteps.
- **Tabular Data**:
  - Includes demographic, clinical, and Viome features.
  - Scaled and one-hot encoded as appropriate.

### Machine Learning Pipeline
- Data preprocessing includes handling missing values, scaling, and sequence normalization.
- Model implemented using **PyTorch** for training and evaluation.
- **Keras** is used for certain preprocessing tasks, like padding sequences.

---

## Project Workflow
1. **Data Preprocessing**:
   - Clean and preprocess image, time-series, and tabular data.
   - Ensure consistency in feature scaling and sequence length for training and testing sets.
2. **Model Implementation**:
   - Utilize a multimodal neural network to process image, time-series, and tabular data streams.
   - Combine outputs from different modalities for lunch calorie prediction.
3. **Evaluation**:
   - Perform an 80/20 train-validation split to evaluate model performance.
   - Metrics like Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) are used to assess accuracy.

---

## Technologies Used
### Frameworks and Libraries
- **PyTorch**: Model implementation and training.
- **Keras**: Data preprocessing.
- **Pandas/Numpy**: Data manipulation and analysis.
- **Scikit-learn**: Scaling and encoding features.
- **Matplotlib**: Visualizations of results and data.

### Tools
- **Jupyter Notebook**: Code development and documentation.
- **Python**: Primary programming language.

---

## Dataset
- **Image Data**: RGB images representing meals, resized to `(224, 224)`.
- **Time-Series Data**: CGM glucose readings sampled every 5 minutes.
- **Tabular Data**: Demographic, clinical, and microbiome features.

### Data Details
- Image, time-series, and tabular data are processed into PyTorch tensors for training and testing.
- Labels represent calorie intake values for supervised learning.

---
