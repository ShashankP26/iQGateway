



 Emotion Classification in Text Data

Overview

This project focuses on developing a machine learning model to classify emotions expressed in short text messages. The model is trained on a dataset labeled with six emotion classes: sadness, joy, love, anger, fear, and surprise. The goal is to accurately identify and categorize emotions to facilitate applications such as mental health monitoring, customer feedback analysis, and sentiment detection.

Dataset

The dataset consists of three main CSV files:
- Training data: Used for training the emotion classification model.
- Validation data: Employed during model training to validate performance.
- Test data: Reserved for final evaluation after model training.

 Data Distribution

To ensure robust evaluation, a balanced distribution of data across validation and test sets was maintained. This included redistributing a portion of the original test set to the validation set.

 Label Distribution

The distribution of emotion labels in the training data is as follows:
- Sadness: 23%
- Joy: 20%
- Love: 17%
- Anger: 15%
- Fear: 13%
- Surprise: 12%

 Data Preprocessing

Text Preprocessing

Text preprocessing involved several steps:
- Stemming: Reducing words to their root form using the Porter Stemmer.
- Stopword Removal: Eliminating common words that do not contribute to the overall meaning.
- Tokenization: Converting text into sequences of tokens.
- Padding: Ensuring all sequences are of uniform length for model input.

Tokenization and Vocabulary

A tokenizer was utilized across all datasets to maintain consistent mapping of words to numerical indices. The vocabulary size handled approximately 16,000 unique words.

 Model Architecture

Neural Network Design

The emotion classification model is designed using TensorFlow's Keras API:
- Embedding Layer: Maps word indices to dense vectors.
- Bidirectional LSTM: Captures contextual dependencies bidirectionally.
- Dropout Layers: Mitigates overfitting by randomly dropping neurons.
- Dense Layer: Outputs probabilities for each emotion class using softmax activation.

 Training and Optimization

The model is optimized using the Adam optimizer with a learning rate of 0.003 and trained for 25 epochs. Training and validation metrics (accuracy, loss) are monitored to ensure optimal model performance.

Model Evaluation

Performance Metrics

During training, the model achieved:
- Training Accuracy: 88%
- Validation Accuracy: 86%
- Training Loss: 0.32
- Validation Loss: 0.38

The final evaluation on the test set resulted in an accuracy of 85%, demonstrating robust generalization to unseen data.

Confusion Matrix Analysis

A confusion matrix provides insights into the model's predictions versus actual labels across all emotion classes, highlighting areas of strength and potential improvement.

 Example Predictions

Randomly selected examples from the test set showcase the model's ability to accurately classify emotions in text messages, indicating reliable performance across diverse instances.

Discussion and Conclusion

 Insights

- Effective preprocessing enhances model performance.
- Bidirectional LSTM effectively captures contextual dependencies.
- Balanced dataset distribution contributes to improved model generalization.

Limitations and Future Directions

- Consideration of advanced neural network architectures (e.g., attention mechanisms).
- Integration of additional features or metadata for richer emotion classification.
- Further fine-tuning of hyperparameters to potentially enhance model metrics.

Conclusion

This project presents a significant advancement in emotion classification from text data, applicable across various real-world scenarios. Continued refinement and exploration of advanced techniques promise increased accuracy and broader applicability in emotion detection tasks.


