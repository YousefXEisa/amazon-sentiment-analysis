import torch
import sys
import time
from tqdm import tqdm
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from src.utils import (
    get_device,
    tokenize,
    load_model_and_tokenizer,
    get_confusion_matrix,
    DEFAULT_MODEL_PATH,
    DATA_PATH,
    BATCH_SIZE
)


device = get_device()

def load_test_data():
    dataset = load_from_disk(DATA_PATH)
    return dataset['test']

def tokenize_test_data(test_data, tokenizer):
    def tokenize_function(example):
        return tokenize(example, tokenizer)

    tokenized_test = test_data.map(tokenize_function, batched=True)
    tokenized_test = tokenized_test.remove_columns(["title", "content"])
    tokenized_test.set_format("torch")
    return tokenized_test

def create_test_dataloader(tokenized_test, tokenizer):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return DataLoader(
        tokenized_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,

    )

def evaluate_model(model, test_loader):
    all_predictions = []
    all_labels = []

    model.eval()
    progress_bar = tqdm(test_loader, desc="Evaluating", file=sys.stdout)

    with torch.no_grad():
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions)

    print(10 * '-' + ' Test results ' + 10 * '-')

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nClassification Report:\n", report)

    return cm



def main():
    print("1. Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(DEFAULT_MODEL_PATH, device)

    print("2. Loading test dataset...")
    test_data = load_test_data()

    print("3. Tokenizing test data...")
    tokenized_test = tokenize_test_data(test_data, tokenizer)

    print("4. Creating DataLoader...")
    test_loader = create_test_dataloader(tokenized_test, tokenizer)

    print(f"Number of batches: {len(test_loader)}")

    print("\n5. Starting Evaluation...")
    start = time.time()
    cm = evaluate_model(model, test_loader)
    end = time.time()
    print(f"Evaluation Time: {end - start:.2f} sec")

    get_confusion_matrix(cm)

if __name__ == "__main__":
    main()