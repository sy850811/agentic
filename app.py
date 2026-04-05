import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from transcribe.pipeline import TranscriptionPipeline

_pipeline: TranscriptionPipeline | None = None


def _get_pipeline() -> TranscriptionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = TranscriptionPipeline()
    return _pipeline


def transcribe(audio_path: str | None) -> tuple[str, str]:
    """Returns (transcript, status)."""
    if not audio_path:
        return "", "Upload an audio file first."
    try:
        pipeline = _get_pipeline()
        result = pipeline.run(audio_path)
        return result, "Done."
    except RuntimeError as exc:
        return "", f"Configuration error: {exc}"
    except Exception as exc:
        return "", f"Error: {exc}"


with gr.Blocks(theme=gr.themes.Soft(), title="Voice Memo Transcriber") as demo:
    gr.Markdown("# Voice Memo Transcriber")
    gr.Markdown(
        "Upload an iOS Voice Memo (`.m4a`, `.mp3`, `.wav`). "
        "Hindi is transcribed in Roman script, English stays in English."
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                type="filepath",
                label="Voice Memo",
            )
            transcribe_btn = gr.Button("Transcribe", variant="primary")
            status_box = gr.Textbox(
                label="Status",
                interactive=False,
                max_lines=2,
            )

        with gr.Column(scale=2):
            transcript_box = gr.Textbox(
                label="Transcript",
                lines=24,
                show_copy_button=True,
                placeholder="Your transcript will appear here...",
            )

    transcribe_btn.click(
        fn=transcribe,
        inputs=audio_input,
        outputs=[transcript_box, status_box],
    )

    # Clear transcript when a new file is uploaded
    audio_input.change(
        fn=lambda _: ("", ""),
        inputs=audio_input,
        outputs=[transcript_box, status_box],
    )


if __name__ == "__main__":
    demo.launch()
