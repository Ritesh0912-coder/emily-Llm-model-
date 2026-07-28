"""
Gradio demo for Emily SLM.

Launch with:
    python slm/frontend/demo.py \
        --model checkpoints/emily-tiny/best \
        --tokenizer checkpoints/emily-tiny/tokenizer.json
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def build_demo(engine: object) -> object:
    """Build and return the Gradio Blocks interface."""
    import gradio as gr

    def _generate(prompt: str, max_tokens: int, temperature: float,
                  top_k: int, top_p: float) -> str:
        return engine.generate(  # type: ignore[attr-defined]
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    def _chat(message: str, history: list[list[str]],
              max_tokens: int, temperature: float) -> str:
        messages = [{"role": "system", "content": "You are Emily, a helpful AI assistant."}]
        for user_msg, asst_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": asst_msg})
        messages.append({"role": "user", "content": message})
        return engine.chat(  # type: ignore[attr-defined]
            messages=messages,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )

    model_name = "Emily SLM"
    try:
        cfg = engine.model.config  # type: ignore[attr-defined]
        model_name = f"Emily SLM — {cfg.name}"
    except Exception:
        pass

    with gr.Blocks(
        title=model_name,
        theme=gr.themes.Soft(primary_hue="purple", secondary_hue="blue"),
        css="""
            .gradio-container { max-width: 900px; margin: 0 auto; }
            .title-text { text-align: center; margin-bottom: 1rem; }
        """,
    ) as demo:
        gr.HTML(f"""
            <div class="title-text">
                <h1>🧠 {model_name}</h1>
                <p style="color: #888;">Custom GPT-style decoder-only transformer built from scratch</p>
            </div>
        """)

        with gr.Tab("💬 Chat"):
            chatbot = gr.ChatInterface(
                fn=_chat,
                title="",
                description="Have a conversation with Emily",
                additional_inputs=[
                    gr.Slider(32, 1024, value=256, step=32, label="Max Tokens"),
                    gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature"),
                ],
            )

        with gr.Tab("✍️ Text Generation"):
            with gr.Row():
                with gr.Column():
                    prompt_box = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt here…",
                        lines=6,
                    )
                    with gr.Row():
                        max_tok = gr.Slider(32, 1024, value=256, step=32, label="Max Tokens")
                        temp = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature")
                    with gr.Row():
                        top_k = gr.Slider(0, 200, value=50, step=5, label="Top-K")
                        top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-P")
                    generate_btn = gr.Button("Generate", variant="primary")

                with gr.Column():
                    output_box = gr.Textbox(label="Generated Text", lines=12)

            generate_btn.click(
                fn=_generate,
                inputs=[prompt_box, max_tok, temp, top_k, top_p],
                outputs=output_box,
            )

        with gr.Tab("ℹ️ Model Info"):
            try:
                cfg = engine.model.config  # type: ignore[attr-defined]
                from slm.utils.device import count_parameters, format_parameters
                n_params = count_parameters(engine.model)  # type: ignore[attr-defined]
                info_text = f"""
| Property | Value |
|----------|-------|
| Model Name | `{cfg.name}` |
| Parameters | **{format_parameters(n_params)}** |
| Vocabulary Size | {cfg.vocab_size:,} |
| Context Length | {cfg.context_length:,} |
| Hidden Dim (d_model) | {cfg.d_model} |
| Attention Heads | {cfg.n_heads} |
| Layers | {cfg.n_layers} |
| FFN Dim | {cfg.d_ff} |
| Activation | {"SwiGLU" if cfg.use_swiglu else "GeLU"} |
| Normalisation | {"RMSNorm" if cfg.use_rms_norm else "LayerNorm"} |
| Position Encoding | {"RoPE" if cfg.use_rope else "Learnable"} |
| Weight Tying | {cfg.tie_embeddings} |
"""
            except Exception:
                info_text = "Model info unavailable."

            gr.Markdown(info_text)

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Emily SLM Gradio Demo")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer JSON")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", default=7860, type=int, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    from slm.utils.logging import setup_logger
    setup_logger("slm", level=logging.INFO)

    from slm.inference.engine import EmilyInferenceEngine
    engine = EmilyInferenceEngine(model_path=args.model, tokenizer_path=args.tokenizer)

    demo = build_demo(engine)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
