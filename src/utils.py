import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MAX_LENGTH = 192
DEFAULT_MODEL_PATH = '../checkpoints/best_model'
DATA_PATH = '../data/amazon_dataset'
BATCH_SIZE = 8

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_and_tokenizer(model_path: str = DEFAULT_MODEL_PATH, device: torch.device = None, quantize: bool = True):
    if device is None:
        device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)

    if quantize and device.type == "cpu":
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

    model.eval()
    return model, tokenizer

def tokenize(example, tokenizer):
    return tokenizer(
        example["title"],
        example["content"],
        truncation=True,
        max_length=MAX_LENGTH
        )

def get_confusion_matrix(cm, save_path="../results/confusion_matrix.png"):
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")