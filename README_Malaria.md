##  Malaria Parasite Classification using CNNs

This repository contains Python code for training a Convolutional Neural Network (CNN) model to classify malaria parasites in blood smear images.

**Project Goal:**

Develop a CNN model to automate malaria parasite classification from blood smear images, potentially aiding in faster and more objective diagnosis.

**Getting Started:**

1. **Clone the repository:**

```bash
git clone https://github.com/your_username/malaria_parasite_classification.git
```

2. **Install dependencies:**

Create a virtual environment and install the required libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

**Data Preparation:**

The script expects your data to be organized into two folders:

- `Parasitized`: Containing images of blood smears with malaria parasites.
- `Uninfected`: Containing images of blood smears without malaria parasites.

**Running the Script:**

```bash
python train_model.py
```

This script will:

- Load the data.
- Preprocess the images.
- Build the CNN model.
- Train the model on the training data.
- Evaluate the model's performance on the validation data.

**Outputs:**

The script will print the training and validation accuracy and loss values during each epoch. It will also save the best performing model based on the validation loss.



