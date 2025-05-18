

# Multimodal Visual Question Answering (VQA) using ABO Dataset

## Overview

This project explores Visual Question Answering (VQA) using the Amazon-Berkeley Objects (ABO) dataset. The task involves curating a single-word answer VQA dataset with multimodal tools, evaluating pretrained models, fine-tuning them using Low-Rank Adaptation (LoRA), and benchmarking performance using accuracy metrics.

## Data Curation

- Source: Amazon Berkeley Objects Dataset (small variant ~3GB)
- Images: We used 15k catalog images (256x256), and created 2-3 questions per image
- Metadata: Product-level metadata from `images.csv.gz`

We used the Google Gemini 2.0 API (`google.generativeai`) to create questions.

The given metadata was cleaned, and product name, features, and tags were extracted for each product. The cleaned metadata and the image were passed to Gemini with the below prompt template. The model's response was parsed and questions were extracted as json.

We used 15K images and created 2-3 questions per image. 
Final dataset size: 37468 questions

**Prompt Template:**
```
Given a product image, generate a set of diverse questions that can be answered solely by looking at the image.
Each question must have a single-word answer (e.g., "red", "shoes", "five", "yes").
Cover a range of question types, including:

- Object recognition (e.g., What product is shown?)
- Attribute detection (e.g., What color is the product?)
- Material/texture recognition (e.g., What material is the product made of?)
- Size/shape recognition (e.g., What is the shape of the product?)
- Brand recognition (e.g., What brand is the product?)
- Counting (e.g., How many items are in the image?)
- Yes/No questions (e.g., Is the product a smartphone?)

Mix easy and challenging questions. Avoid subjective or ambiguous questions.

Use this product information for reference but only generate questions that can be answered from the image alone.

Product Information:
[METADATA]

Output Format:
For each product image, return a list of 3-4 questions and their single-word answers. Respond in this format (do not include any extra information):

[
    {
        "question": "What product is shown?",
        "answer": "Laptop"
    },
    {
        "question": "What color is the product?",
        "answer": "Black"
    },
    {
        "question": "Is the logo visible?",
        "answer": "Yes"
    },
    {
        "question": "What material is the product made of?",
        "answer": "Plastic"
    },
    {
        "question": "Is the product in a box?",
        "answer": "No"
    }
]
```



## Baseline Evaluation

We used a subset of the created questions for baseline evaluation.

Train data size - 30523 questions

Test data size - 6945 questions

We evaluated both the baseline and fine-tuned models on only the test data to ensure that the model had not seen the questions during training. 
 
### Models Used
- **BLIP (Bootstrapping Language-Image Pretraining)**
   BLIP is a transformer-based vision-language model that integrates a vision encoder (ViT) and a text decoder, trained on large-scale image-text pairs for tasks like image captioning, visual question answering, and retrieval. Nearly 385M parameters.
- **ViLT (Vision-and-Language Transformer)**
   ViLT is a lightweight model that removes the convolutional visual backbone and directly processes image patches using transformers. Nearly 86M parameters.

### Evaluation metrics
- Compared predicted answers with ground truth using:
  - **Exact Match Accuracy**: Proportion of predictions that exactly match the ground truth word.
  - **BERTScore**: Semantic similarity metric using contextual embeddings from BERT.
  - **BLEU Score**: Measures the overlap of n-grams between predicted and reference answers.
  - **METEOR Score**: Considers synonymy, stemming, and word order for better alignment with human judgment.

  
| Model         | Exact Match Accuracy | BERTScore | BLEU Score | METEOR Score |
|---------------|----------------------|-----------|------------|--------------|
| BLIP baseline       | 0.49                | 0.9792     | 0.4953     | 0.2636        |
|Vilt baseline      | 0.471                | 0.977    | 0.47      | 0.2528    |

### Observation
- BLIP baseline outperformed ViLT across all metrics. This is because BLIP is larger and more powerful than ViLT.

## Fine-Tuning using LoRA

### Model Selection
- Based on the baseline evaluation results, we chose to fine-tune BLIP since it performed better than ViLT. We chose to not fine-tune BLIP-2 since it is a large model with 2.7 B parameters.
- We experimented fine-tuning with various LoRA configurations

### Training Details

- Trained and tested 2 configurations for LoRA fine-tuning.
- Hyperparameters such as batch size were experimented with.
- The baseline models and the fine-tuned models were evaluated only on test data. 

Best model hyperparameters:

| Parameter          | Value    |
|--------------------|---------------------|
| Model              | BLIP            |
| LoRA Rank          | 16                   |
| Epochs             | 8                 |
| Batch Size         | 12                  |
| Learning Rate      | 5e-5  with learning rate scheduler |

### LoRA configurations used

| Config   | Rank (`r`) | Alpha | Dropout | Target Modules       | Bias |
| -------- | ---------- | ----- | ------- | -------------------- | ---- |
| Config 1 | 8          | 16    | 0.1     | `["query", "value"]` | none |
| Config 2 | 16         | 32    | 0.2     | `["query", "value"]` | none |

Config 2 gave a better accuracy. 

### Results after fine-tuning

| Model         | LoRA config | Batch Size | Exact Match Accuracy | BERTScore | BLEU Score | METEOR Score |
|---------------|-----|-------|----------|-----------|------------|--------------|
| BLIP fine-tuned  | config 1     | 12 | 0.777                | 0.9753   | 0.7776      | 0.3925        |
|  BLIP fine-tuned | config 2  | 12 | 0.779                 | 0.9753     | 0.779     | 0.3933         |
| BLIP fine-tuned | config 2  | 8 | 0.7758 | 0.9753 | 0.775 | 0.3921 |

## Observations

1. **Exact Match Accuracy** improved by nearly **29%**, showing that the model was better at generating the exact expected word after fine-tuning.

2. **BERTScore** was high before fine-tuning and did not change much after fine-tuning as well. It evaluates the semantic similarity between the predicted and reference answers using contextual embeddings from BERT. This shows that both baseline and fine-tuned models produced semantically relevant answers, even if the exact token or phrasing changed. BERTScore is less sensitive to small token-level changes, so it’s not a strong indicator for slight improvements after fine-tuning.


3. **BLEU Score** and **METEOR Score**, which measure fluency and partial match/synonymy, increased significantly, indicating both better wording and alignment with the expected output.

4. LoRA Config 2 slightly outperformed Config 1 across all metrics. The increase in `r`, `alpha`, and `dropout` in Config 2 could have helped to avoid overfitting by regularizing better. However, the improvements are marginal. 

5. Changing the batch size did not change the accuracy by much.

6. ![Screenshot from 2025-05-18 17-27-29](https://github.com/user-attachments/assets/473fb584-bd2d-4bc4-8b8d-2cf7aa6b582a)

7. On evaluating the model after training for 1 epoch, we observed 75% exact match accuracy. After 8 epochs, we observed 79%. This is justified as the loss did not decrease by much after a few epochs. After 8 epochs, the the validation loss started increasing.

8. ![Screenshot from 2025-05-18 15-41-20](https://github.com/user-attachments/assets/d19da32d-cf82-447c-8cf8-ca4beb6f6569)

9. The plot shows the accuracy of the model with different types of questions. This analysis shows that VQA performance is highly dependent on clarity of features relevant to the question type:

   - Color is not ambiguous, and directly learnable.
   - Brand recognition is harder. The model might benefit from including OCR modules for better brand and material recognition.

10. We tried evaluating initially on smaller test data set. The accuracy increased when we evaluated it on the entire test set.

### Quantization and KV caching 

- The interpretation duration 

## Challenges 

- We initially used only the metadata without the image for dataset curation. We observed that the model hallucinated and created questions that could not be answered by simply looking at the image. Baseline evaluation gave a low accuracy. We resolved this by giving the image as an input during dataset curation.
- We faced API rate limits while using Gemini, especially since we gave the image as an input. We resolved this by using several API keys in rotation.


## Files

- `dataset_preparation.ipynb`: Full data curation process.
- `dataset.csv`: CSV file with image path, question and answer.
- `baseline.ipynb`: baseline and fine-tuned model evaluation.
- `fine_tuning.ipynb`: Fine-tuning using LoRA.
- `run_inference.py`: Inference script to run on new images.
- `requirements.txt`: dependencies to run the code.
