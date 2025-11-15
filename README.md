# Twitter Sentiment Analysis Using LSTM & GloVe

A deep learning project for classifying Twitter messages into **hate speech** vs **non-hate** using an LSTM-based neural network and **GloVe word embeddings**.  
The project includes full preprocessing, embedding conversion, sequence padding, model training, evaluation, and saving the best-performing model.

---

Features
- Full text preprocessing (tokenizing, lowercasing, lemmatization)
- GloVe 50d word embedding integration
- Variable-length sequence handling with dynamic padding
- Deep LSTM architecture with multi-layer Dropout
- Class imbalance handling using `class_weight`
- Validation & test split following best practices
- Model checkpointing to save the best model
- Detailed evaluation using precision, recall, and F1-score

---

 Model Architecture
- **Input Shape:** `(57, 50)`
- **Embedding:** GloVe 6B 50d
- **Layers:**
  - 4× LSTM layers (50 units, `return_sequences=True`)
  - Dropout(0.2) after every LSTM layer
  - Flatten + Dense(1, activation='sigmoid')
- **Loss:** Binary Crossentropy  
- **Optimizer:** Adam (lr = 0.0001)  
- **Metrics:** Accuracy, AUC

---
