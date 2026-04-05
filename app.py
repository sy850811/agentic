import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from transcribe.open_brain import capture
from transcribe.pipeline import TranscriptionPipeline

_pipeline: TranscriptionPipeline | None = None


def _get_pipeline() -> TranscriptionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = TranscriptionPipeline()
    return _pipeline


def transcribe(audio_path: str | None) -> tuple[str, str]:
    if not audio_path:
        return "", "Record or upload audio first."

    try:
        result = _get_pipeline().run(audio_path)
    except Exception as exc:
        return "", f"Transcription error: {exc}"

    try:
        capture(result, {"source": "voice"})
    except Exception as exc:
        return result, f"Transcribed, but Open Brain capture failed: {exc}"

    return result, "Saved to Open Brain."


with gr.Blocks(theme=gr.themes.Soft(), title="Open Brain Recorder") as demo:
    gr.Markdown("# Open Brain Recorder")
    gr.Markdown("Record or upload audio. Transcribes Hindi + English and saves to Open Brain.")

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Record or Upload",
            )
            save_btn = gr.Button("Transcribe & Save", variant="primary")
            status_box = gr.Textbox(label="Status", interactive=False, max_lines=2)

        with gr.Column(scale=2):
            transcript_box = gr.Textbox(
                label="Transcript",
                lines=24,
                show_copy_button=True,
                placeholder="Your transcript will appear here...",
            )

    save_btn.click(
        fn=transcribe,
        inputs=audio_input,
        outputs=[transcript_box, status_box],
    )
    audio_input.change(
        fn=lambda _: ("", ""),
        inputs=audio_input,
        outputs=[transcript_box, status_box],
    )


if __name__ == "__main__":
    demo.launch()
