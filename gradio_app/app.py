import gradio as gr
from src.inference import SentimentPredictor

predictor = SentimentPredictor()


def analyze_sentiment(title: str, content: str):
    try:
        result = predictor.predict(title=title, content=content)

        return (
            f"### Predict: **{result['label']}** ({round(result['confidence'] * 100, 1)}%)",
            result["probabilities"]
        )
    except ValueError as ve:
        return f"{str(ve)}", None
    except Exception as e:
        return f"Error: {str(e)}", None


with gr.Blocks(title="Amazon Review Sentiment") as demo:
    gr.Markdown(
        """
        # 🛒 Amazon Review Sentiment Analysis
        Predict sentiment using fine-tuned **RoBERTa**.

        > **Note:** You can provide either a **Title**, **Content**, or **both**.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            title_input = gr.Textbox(
                label="Review Title (Optional)",
                placeholder="e.g., Great purchase!"
            )
            content_input = gr.Textbox(
                label="Review Content (Optional)",
                placeholder="e.g., Battery life is good, but shipping took long.",
                lines=4
            )
            submit_btn = gr.Button("Analyze", variant="primary")

        with gr.Column(scale=1):
            result_output = gr.Markdown(label="Output")
            label_output = gr.Label(label="Confidence & Probabilities", num_top_classes=2)

    gr.Examples(
        examples=[
            ["Great value for money", "The battery life is exceptional and performance is smooth."],
            ["Disappointed", "The product arrived broken and stopped working after two days."],
            ["", "Decent product, does the job perfectly."]
        ],
        inputs=[title_input, content_input],
        outputs=[result_output, label_output],
        fn=analyze_sentiment,
        cache_examples=False
    )

    submit_btn.click(
        fn=analyze_sentiment,
        inputs=[title_input, content_input],
        outputs=[result_output, label_output]
    )

if __name__ == "__main__":
    demo.launch()