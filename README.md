# News Topic Classifier Using BERT

This project fine-tunes a `bert-base-uncased` transformer model on the AG News dataset to classify news headlines into one of four categories:

-  World
-  Sports
-  Business
-  Science/Technology

---

## Demo
 [Live Gradio App](#)  
_(Link will be available after deploying in Kaggle or locally with `share=True`)_

---

## Project Objectives

- Fine-tune a pretrained BERT model using Hugging Face Transformers
- Classify news headlines from the AG News dataset
- Evaluate model performance using Accuracy and F1-score
- Deploy using Gradio for real-time predictions

---

## Dataset

**AG News**  
- 120,000 training samples  
- 7,600 test samples  
- Labels: World, Sports, Business, Sci/Tech

Dataset source: Hugging Face Datasets  

```python
from datasets import load_dataset
load_dataset("ag_news")
```

## Tech Stack

- BERT from Hugging Face Transformers
- AG News from Hugging Face Datasets
- PyTorch for model training
- Gradio for UI deployment


## Evaluation Metrics

```bash
Accuracy: 0.94
F1-Score (weighted): 0.94
```

## Project Structure

```bash
News-Topic-Classifier-BERT/
│
├── News-Topic-Classifier-BERT.ipynb          # Model training and evaluation, also contains Gradio Interface
├── saved_model/                              # Final fine-tuned model
├── results/                                  # Training logs
└── README.md                                 # Project documentation
```

## Usage
1. Install dependencies

```bash
pip install transformers datasets gradio scikit-learn torch
```

2. Run the notebook file
   
3. Run Gradio App inside the notebook:

```python
interface.launch(share=True)
```

## Acknowledgements
- Hugging Face for Transformers and Datasets
- AG News dataset contributors
- Gradio for simplified deployment
