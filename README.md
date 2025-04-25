# 🎓 Bachelor of IT – Academic Project Portfolio

Welcome to my project portfolio! This repository showcases practical coursework completed as part of my Bachelor of Information Technology at Macquarie University. It focuses on real-world data science, deep learning, and NLP challenges, with a strong emphasis on explainability, tuning, and ethical considerations.

---

## 📊 COMP2200: Data Science Projects

### 📌 1. Heart Disease Analysis
- **Goal**: Predict heart disease using logistic regression and clustering.
- **Dataset**: UCI Heart Disease Dataset
- **Key Features**: Age, Sex, Chest Pain, BP, Cholesterol, etc.
- **Accuracy**: ~83.19%
- **Highlights**:
  - Correlation matrix for feature importance
  - Gender-based risk analysis
  - K-means clustering for risk group identification

### 📌 2. Mobile Price Prediction
- **Goal**: Predict mobile phone price range
- **Techniques**: Logistic Regression, KNN, GridSearchCV
- **Best Accuracy**:
  - KNN: ~93% (train), ~89% (test)
  - Logistic Regression: ~89% (train/test)

### 📌 3. E-commerce User Ratings Analysis
- **Goal**: Model user product ratings
- **Methods**: Linear Regression, RMSE evaluation
- **Key Insights**:
  - Negative correlation between product category and rating
  - More data improved model accuracy
  - Strong focus on ethical implications of data presentation

---

## 🤖 COMP3420: AI for Text and Vision

### 📌 1. Image Processing
- **Functions Built**:
  - `light_pixels`: Mask pixels above intensity threshold
  - `histogram`: Manual per-channel histogram (no external libs)

### 📌 2. MNIST Digit Classifier & Hyperparameter Tuning
- **Goal**: Classify handwritten digits using deep neural networks
- **Tools**: Keras, Keras Tuner (Bayesian Optimization)
- **Best Accuracy**: ~97.94%
- **Hyperparameters Tuned**: Hidden layers, size, dropout

### 📌 3. Intel Image Scene Classification
- **Dataset**: 6 outdoor scenes (forest, sea, buildings, etc.)
- **Models**:
  - Baseline DNN
  - CNN with Conv2D + MaxPooling2D
  - Transfer Learning with MobileNet (ImageNet)
- **Highlights**:
  - MobileNet outperformed other models
  - Visual error analysis with confusion matrices

### 📌 4. Text Classification & NLP Tasks
- **Libraries**: NLTK, Scikit-learn
- **Key Tasks**:
  - `topN_pos`: Extract top N nouns (POS tagging)
  - `topN_2grams`: Frequent bigrams with/without stemming
  - `sim_tfidf`: Cosine similarity on QA pairs (TF-IDF)
- **Applications**: Question similarity, keyword detection, content filtering

