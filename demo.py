import gradio as gr
from src.model_inference import ABSAPipeline

pipeline = ABSAPipeline(
    ext_model_path="./train/saved_models/aspect_extractor",
    cls_model_path="./train/saved_models/sentiment_classifier"
)
print("Tải mô hình thành công!")

def analyze_review(text):
    results = pipeline.predict(text)
    if not results:
        return [["Không có", "Không phát hiện đối tượng", "Không có"]]
    return results

with gr.Blocks(theme=gr.themes.Soft()) as demo: # type: ignore
    gr.Markdown("# Phân Tích Cảm Xúc Theo Khía Cạnh")
    gr.Markdown("Nhập một đoạn đánh giá để xem cách hệ thống tách câu, trích xuất **Đối tượng (Aspect)** và phân loại **Cảm xúc (Sentiment)**.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                lines=8, 
                placeholder="Ví dụ: The screen is amazing but battery life is terrible. I also think the price is way too high for what it offers...", 
                label="Đoạn đánh giá"
            )
            submit_btn = gr.Button("Phân Tích", variant="primary")
            
        with gr.Column(scale=1.5): # type: ignore
            output_df = gr.Dataframe(
                headers=["Trích xuất từ Câu", "Đối tượng (Aspect)", "Cảm xúc (Sentiment)"],
                label="Kết quả Phân tích",
                interactive=False,
                wrap=True
            )
            
    submit_btn.click(fn=analyze_review, inputs=input_text, outputs=output_df)

    gr.Examples(
        examples=[
            ["Great food, cozy atmosphere, and very friendly staff."],
            ["The food was excellent but the service was terribly slow."],
            ["The food is not good"],
            ["The waiter forgot our order."],
            ["the food was sooo gooood!!!"],
            ["The waiter was so attentive that we never saw him."],
            ["The service could be better."],
            ["I upgraded to a MacBook Pro 15-inch recently. The Retina display is absolutely gorgeous and the keyboard feels great to type on. However, the fan gets extremely loud when I train deep learning models, and the customer support was useless when I asked about it."]
        ],
        inputs=input_text
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)