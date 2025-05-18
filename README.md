

# Multimodal Visual Question Answering (VQA) using ABO Dataset

## Overview

This project explores Visual Question Answering (VQA) using the Amazon-Berkeley Objects (ABO) dataset. The task involves curating a single-word answer VQA dataset with multimodal tools, evaluating pretrained models, fine-tuning them using Low-Rank Adaptation (LoRA), and benchmarking performance using accuracy metrics.

## 1. Data Curation

### Tools Used
- Google Gemini 2.0 API (`google.generativeai`)
- Python libraries: `os`, `cv2`, `json`, `gzip`, `time`, `random`
- Environment: Google Colab and Kaggle 

### Dataset
- Source: Amazon Berkeley Objects Dataset (small variant ~3GB)
- Images: 15k catalog images (256x256) with 2-3 questions per image
- Metadata: Product-level metadata from `images.csv.gz`
### Preprocessing Steps
1. Downloaded and extracted the ABO small dataset and associated metadata.
2. Mapped image file paths to product IDs using the metadata CSV.
3. The given metadata was cleaned and product name, features, and tags were extracted for each product
4. Cleaned metadata to use as additional context during prompt generation.

### Prompt Engineering
The Gemini API was used to generate diverse single-word QA pairs per image using the following structure:

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

## 2. Baseline Evaluation

### Models Used
- **BLIP (Bootstrapping Language-Image Pretraining)**
   BLIP is a transformer-based vision-language model that integrates a vision encoder (ViT) and a text decoder, trained on large-scale image-text pairs for tasks like image captioning, visual question answering, and retrieval. Nearly 385M parameters.
- **ViLT (Vision-and-Language Transformer)**
   ViLT is a lightweight model that removes the convolutional visual backbone and directly processes image patches using transformers. Nearly 86M parameters.

### Evaluation Setup
- Loaded the curated VQA dataset (image, question, answer) ie testing data set.
- Performed inference with BLIP and Vilt without any fine-tuning .
- Compared predicted answers with ground truth using:
  - **Exact Match Accuracy**: Proportion of predictions that exactly match the ground truth word.
  - **BERTScore**: Semantic similarity metric using contextual embeddings from BERT.
  - **BLEU Score**: Measures the overlap of n-grams between predicted and reference answers.
  - **METEOR Score**: Considers synonymy, stemming, and word order for better alignment with human judgment.

### Observation
- BLIP baseline outperformed ViLT across all metrics,this is due to the larger model size of the BLIP.

## 3. Fine-Tuning using LoRA

### Model Selection
- Based on the baseline evaluation results, we chose to fine-tune BLIP. We chose BLIP since it performed better than ViLT. We chose to not fine-tune BLIP-2 since it is a large model with 2.7 B parameters.
- We experimented fine-tuning with various LoRA configurations

### Training Details - best model
| Parameter          | Value (example)     |
|--------------------|---------------------|
| Model              | BLIP            |
| LoRA Rank          | 16                   |
| Epochs             | 10                  |
| Batch Size         | 12                  |
| Learning Rate      | 5e-5                |
| Hardware           | Kaggle Dual T4 GPUs (32GB)

### Configurations tested

| Config   | Rank (`r`) | Alpha | Dropout | Target Modules       | Bias |
| -------- | ---------- | ----- | ------- | -------------------- | ---- |
| Config 1 | 8          | 16    | 0.1     | `["query", "value"]` | none |
| Config 2 | 16         | 32    | 0.2     | `["query", "value"]` | none |

### Observations
- Trained and tested 2 configurations for the LoRa fine-tuning 
- Hyper parameters such as batch size were set different to try out different settings
- The test data was a part from the whole data.We used 30523 questions in train and 6945 questions in test. Total data set size was 15k and has 2-3 question per image.
- The baseline models and the fine-tuned models were evaluated only on test data. 

| Model         | Exact Match Accuracy | BERTScore | BLEU Score | METEOR Score |
|---------------|----------------------|-----------|------------|--------------|
| BLIP baseline       | 0.49                | 0.9792     | 0.4953     | 0.2636        |
|Vilt baseline      | 0.471                | 0.977    | 0.47      | 0.2528    |
| BLIP fine-tuned  config1     | 0.777                | 0.9753   | 0.7776      | 0.3925        |
|  BLIP fine-tuned config 2  | 0.779                 | 0.9753     | 0.779     | 0.3933         |
| BLIP fine-tuned different <br/>batch size and optimiser | 0.7758 | 0.9753 | 0.775 | 0.3921 |

1. **BERTScore** evaluates the **semantic similarity** between the predicted and reference answers using contextual embeddings from BERT. Since the fine-tuned models were already generating **semantically correct answers** (e.g., synonyms or similar meanings).

   This shows that both baseline and fine-tuned models produced **semantically relevant answers**, even if the exact token or phrasing changed. BERTScore is less sensitive to small token-level changes, so it’s not a strong discriminator for slight improvements post fine-tuning.

2. **Exact Match Accuracy** improved by nearly **29%**, showing that the model was better at generating the exact expected word.

3. **BLEU Score** and **METEOR Score**, which measure fluency and partial match/synonymy, increased significantly, indicating both better **wording** and **alignment** with the expected output.

4. Config 2 slightly outperformed Config 1 across all metrics. The increase in `r`, `alpha`, and `dropout` in Config 2 likely enabled the model to learn richer representations and avoid overfitting by regularizing better. This suggests that higher-capacity LoRA layers (larger `r`, `alpha`) and higher dropout can help with fine-tuning in low-resource settings. However, the improvements are marginal. 

5. Using different batch size ie 12, 8 and optimizer didnt effect the outputs much.

6. plot![](/home/iiitb/Pictures/Screenshots/Screenshot from 2025-05-18 17-27-29.png)

7. For the 1st epoch we got 75% exact match accuracy when we evaluated. The accuracy didnot improve much after few epochs and after 8 epochs the validation loss started increasing.

8. <img src="/home/iiitb/.config/Typora/typora-user-images/image-20250518154227228.png" alt="image-20250518154227228" style="zoom:50%;" />

9. This analysis shows that VQA performance is highly dependent on visual saliency and clarity of features relevant to the question type:

   - Color is often unambiguous and directly learnable.
   - Brand recognition and material inference are harder, sometimes requiring textual OCR or more fine-grained features.
   - The model might benefit from multitask training or joint vision-text understanding (e.g., including OCR modules) for better brand and material recognition.

10. We tried evaluating initially on smaller test data set size and gradually increases it and observed the accuracy increased.

### Quantization and KV caching 

- The interpretation duration 

### Challenges Encountered

- API rate limits during data generation resolved via key rotation.
- Only using the meta data without the image led to bad outputs where the model hallucinate

## Files Included

- `dataset_preparation.ipynb`: Full data curation process.
- `dataset_fe.json`: Final dataset with image-to-QA mappings.
- `fine_tuning.ipynb`: Fine-tuning script using LoRA (if provided).
- `run_inference.py`: Inference script to run on new images (optional).
