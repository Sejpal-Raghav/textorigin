import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_ID = "humarin/chatgpt_paraphraser_on_T5_base"


def paraphrase_texts(texts: list[str], batch_size: int = 8) -> list[str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(device)
    model.eval()

    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Paraphrasing"):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(decoded)

    return results


def generate_paraphrased_class(
    ai_df: pd.DataFrame, sample_size: int = 5000, seed: int = 42
) -> pd.DataFrame:
    sample = ai_df.sample(n=min(sample_size, len(ai_df)), random_state=seed)
    texts = sample["text"].tolist()
    paraphrased = paraphrase_texts(texts)
    return pd.DataFrame({"text": paraphrased, "source": "ai_paraphrased"})
