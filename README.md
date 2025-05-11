# VR_Project2_IMT2022018_523_572

1. Dataset creation

   Code can be found at `dataset_preparation.ipynb`.

2. Baseline evaluation

   Code can be found at `baseline.ipynb`.

   From our dataset, we chose a subset of x questions for baseline evaluation.
   We chose to evaluate [BLIP](https://huggingface.co/Salesforce/blip-vqa-base) and [ViLT](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa)

  | Model | Model Size | Exact Match F1 Score | BLEU Score | BERTScore | METEOR Score |
  |-------|------------|----------------------|------------|-----------|--------------|
  | BLIP  |   385 M    |                      |            |           |              |
  | ViLT  |            |                      |            |           |              |


3. Fine-tuning

   Code can be found at `fine-tuning.ipynb`.

   Based on the baseline evaluation results, we chose to fine-tune BLIP. We chose BLIP since it performed better than ViLT. We chose to not fine-tune BLIP-2 since it is a large model with 2.7 B parameters.

   We experimented fine-tuning with various LoRA configurations. 
  

5. Performance after fine-tuning
   
   
