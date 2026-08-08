import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
from src.utils import load_model_and_tokenizer, get_device, MAX_LENGTH

HF_REPO_ID = "YousefXEisa/amazon-roberta-sentiment"

class SentimentPredictor:
    def __init__(self, model_path: str= HF_REPO_ID):
        self.device = get_device()
        self.model, self.tokenizer = load_model_and_tokenizer(model_path, self.device)
        self.labels_map: Dict[int, str] = {0: 'Negative', 1: 'Positive'}


    def predict(self, *, title: Optional[str] = None, content: str = "") -> Dict[str, Any]:
        title_str = title.strip() if title is not None else ""
        content_str = content.strip() if content is not None else ""

        if not title_str and not content_str:
            raise ValueError('At least title or content must be provided.')

        text = f"{title_str} {content_str}".strip()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=-1)[0]

        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()

        return {
            'label': self.labels_map[predicted_class],
            'confidence': round(confidence, 4),
            'probabilities': {
                'Negative score': round(probabilities[0].item(), 4),
                'Positive score': round(probabilities[1].item(), 4)
            }
        }

if __name__ == "__main__":
    print("--- Testing SentimentPredictor ---")
    predictor = SentimentPredictor()

    while True:
        print("\n----- Enter Review Details -----")
        title_input = input("Title (optional, press Enter to skip): ").strip()
        content_input = input("Content: ").strip()

        if not title_input and not content_input:
            print("You must provide at least a title or content!")
            continue

        result = predictor.predict(title=title_input, content=content_input)
        print("\nResult:")
        print(result)

        choice = input('\nPress Enter to continue or "q" to quit: ').strip().lower()
        if choice == 'q':
            break