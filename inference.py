
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import os
import zipfile
import gdown

from transformers import BlipProcessor, BlipForQuestionAnswering
from peft import PeftModel, PeftConfig

def download_and_extract_model(drive_file_id, extract_to="./blip_lora"):
    url = f"https://drive.google.com/uc?id={drive_file_id}"
    zip_path = "model.zip"

    if not os.path.exists(extract_to):
        print(f" Downloading model from Google Drive (ID: {drive_file_id})...")
        gdown.download(url, zip_path, quiet=False)
        print(" Extracting model...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f" Model extracted to: {extract_to}")
    else:
        print(f" Model already exists at: {extract_to}")

def main():
    parser = argparse.ArgumentParser(description="Run BLIP LoRA-based VQA inference")
    parser.add_argument('--image_dir', type=str, required=True, help='Path to input images')
    parser.add_argument('--csv_path', type=str, required=True, help='CSV with image_name, question, and answer')
    args = parser.parse_args()

    adapter_path = "./blip_lora"
    drive_file_id = "15Ou8JV6GmPavu5kTtinAwDBPWoiJDB7V" 
    download_and_extract_model(drive_file_id, adapter_path)

    df = pd.read_csv(args.csv_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    peft_config = PeftConfig.from_pretrained(adapter_path)
    processor = BlipProcessor.from_pretrained(peft_config.base_model_name_or_path)
    base_model = BlipForQuestionAnswering.from_pretrained(peft_config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, adapter_path).to(device)
    model.eval()

    predictions = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_path = os.path.join(args.image_dir, row['image_name'])
        question = str(row['question'])

        try:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(image, question, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(**inputs)
            answer = processor.tokenizer.decode(output[0], skip_special_tokens=True)
            answer = answer.split()[0].lower()
        except Exception:
            answer = "error"

        predictions.append(answer)

    df["generated_answer"] = predictions

    # Only keep necessary columns for evaluation
    if 'answer' in df.columns:
        df = df[['image_name', 'question', 'answer', 'generated_answer']]
    else:
        df['answer'] = 'unknown'

    df.to_csv("results.csv", index=False)
    print(" Inference complete. Saved to results.csv")

if __name__ == "__main__":
    main()
