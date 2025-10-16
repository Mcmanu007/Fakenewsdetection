🗞️ Fake News Detection Project

A machine learning project that uses Natural Language Processing techniques to classify news articles as either real or fake using logistic regression and TF-IDF vectorization.

📋 Overview

This project implements a binary classification system to detect fake news articles. The model uses text preprocessing, TF-IDF vectorization, and logistic regression to achieve accurate classification of news content.

📊 Dataset

- 🔗 Source: Kaggle dataset by vishakhdapat
- 📁 Dataset: fake-news-detection
- 📄 File: fake_and_real_news.csv
- 🏷️ Features: Text content and binary labels (Real/Fake)

📂 Project Structure

```
fakenew.ipynb          # Main Jupyter notebook with complete implementation
README.md              # Project documentation
Dataset/               # Downloaded from Kaggle (cached locally)
    fake_and_real_news.csv
```

🔬 Methodology

1. 📥 Data Loading and Exploration
- Downloads dataset using kagglehub
- Loads data using pandas
- Performs initial data exploration and visualization

2. 🧹 Data Preprocessing
- Duplicate Removal: Identifies and removes duplicate entries
- Null Value Check: Ensures data quality
- Class Distribution Analysis: Visualizes balance between real and fake news

3. 📝 Text Preprocessing
The preprocess_text() function performs:
- Lowercasing: Converts all text to lowercase
- Special Character Removal: Removes non-alphabetic characters
- Tokenization: Splits text into individual words
- Stopword Removal: Removes common English stopwords
- Short Word Filtering: Removes words with length less than or equal to 1

4. ⚙️ Feature Engineering
- TF-IDF Vectorization with parameters:
  - max_features=1000: Limits vocabulary size
  - min_df=2: Ignores terms appearing in fewer than 2 documents
  - max_df=0.8: Ignores terms appearing in more than 80% of documents
  - ngram_range=(1,2): Uses unigrams and bigrams
  - sublinear_tf=True: Applies sublinear scaling

5. 🤖 Model Training
- Algorithm: Logistic Regression
- Parameters:
  - max_iter=1000: Maximum iterations for convergence
  - random_state=42: For reproducibility
  - multi_class='multinomial': Multinomial approach
  - solver='lbfgs': Limited-memory BFGS solver

6. 📈 Model Evaluation
- Train-Test Split: 80-20 split with stratification
- Metrics:
  - Accuracy Score
  - Classification Report (Precision, Recall, F1-score)
  - Confusion Matrix with heatmap visualization

📦 Dependencies

```python
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

🛠️ Installation and Setup

1. 💻 Install required packages:
```bash
pip install kagglehub pandas numpy matplotlib seaborn scikit-learn nltk
```

2. 📚 Download NLTK data:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
```

3. ▶️ Run the notebook:
   - Open fakenew.ipynb in Jupyter Notebook or JupyterLab
   - Execute cells sequentially

🚀 Usage

1. 📥 Data Download: The notebook automatically downloads the dataset from Kaggle
2. 🔄 Preprocessing: Text data is cleaned and prepared for modeling
3. 🎯 Training: The logistic regression model is trained on preprocessed text
4. 📊 Evaluation: Model performance is assessed using multiple metrics
5. 📈 Visualization: Results are displayed through confusion matrix heatmap

✨ Key Features

- 🔄 Automated Data Pipeline: From raw text to trained model
- 🧹 Comprehensive Preprocessing: Handles text cleaning and normalization
- ⚙️ Feature Engineering: TF-IDF vectorization with optimized parameters
- 📊 Model Evaluation: Multiple metrics for performance assessment
- 📈 Visualization: Clear plots for data distribution and model performance

📊 Results

The model provides:
- 🎯 Accuracy score on test data
- 📋 Detailed classification report with precision, recall, and F1-scores
- 🔥 Confusion matrix visualization for performance analysis

🔮 Future Improvements

- 🧠 Experiment with other algorithms (Random Forest, SVM, Neural Networks)
- 📝 Implement advanced text preprocessing (stemming, lemmatization)
- ✅ Add cross-validation for more robust evaluation
- 🔍 Include feature importance analysis
- 🌐 Deploy model as a web application

👨‍💻 Author

Machine Learning Project - Fake News Detection

📄 License

This project is for educational purposes.
