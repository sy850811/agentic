import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from transcribe.open_brain import capture
from transcribe.pipeline import TranscriptionPipeline
from transcribe.voice_chat import get_response, text_to_speech

_pipeline: TranscriptionPipeline | None = None


def _get_pipeline() -> TranscriptionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = TranscriptionPipeline()
    return _pipeline


# ---------------------------------------------------------------------------
# Tab 1: Transcription
# ---------------------------------------------------------------------------

def transcribe(audio_path: str | None) -> tuple[str, str]:
    if not audio_path:
        return "", "Upload an audio file first."
    try:
        result = _get_pipeline().run(audio_path)
        capture(result, {"source": "voice_memo", "type": "transcript"})
        return result, "Done. Saved to Open Brain."
    except RuntimeError as exc:
        return "", f"Configuration error: {exc}"
    except Exception as exc:
        return "", f"Error: {exc}"


# ---------------------------------------------------------------------------
# Tab 2: Voice Chat
# ---------------------------------------------------------------------------

# History stored as list of {"role": ..., "content": ...} dicts (OpenAI format).
# Gradio Chatbot displays list of [user_text, assistant_text] pairs.

def chat_turn(
    audio_path: str | None,
    history_state: list[dict],
    chat_display: list[list[str | None]],
) -> tuple[list[list[str | None]], list[dict], str | None, str]:
    """
    Returns (chat_display, history_state, audio_path, status).
    """
    if not audio_path:
        return chat_display, history_state, None, "Record something first."

    try:
        # 1. Transcribe user audio
        user_text = _get_pipeline().run(audio_path)

        # 2. Get GPT-4o response
        assistant_text = get_response(user_text, history_state)

        # 3. TTS
        audio_out = text_to_speech(assistant_text)

        # 4. Capture both turns to Open Brain
        capture(user_text, {"source": "voice_chat", "role": "user"})
        capture(assistant_text, {"source": "voice_chat", "role": "assistant"})

        # 5. Update state
        history_state = history_state + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
        chat_display = chat_display + [[user_text, assistant_text]]

        return chat_display, history_state, audio_out, "Done. Saved to Open Brain."

    except RuntimeError as exc:
        return chat_display, history_state, None, f"Configuration error: {exc}"
    except Exception as exc:
        return chat_display, history_state, None, f"Error: {exc}"


def clear_chat() -> tuple[list, list, None, str]:
    return [], [], None, ""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(theme=gr.themes.Soft(), title="Voice Memo Transcriber") as demo:
    gr.Markdown("# Voice Memo Transcriber")

    with gr.Tabs():

        # ── Tab 1: Transcribe ────────────────────────────────────────────
        with gr.Tab("Transcribe"):
            gr.Markdown(
                "Upload an iOS Voice Memo (`.m4a`, `.mp3`, `.wav`). "
                "Hindi is transcribed in Roman script, English stays in English."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(type="filepath", label="Voice Memo")
                    transcribe_btn = gr.Button("Transcribe", variant="primary")
                    t_status = gr.Textbox(label="Status", interactive=False, max_lines=2)
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
                outputs=[transcript_box, t_status],
            )
            audio_input.change(
                fn=lambda _: ("", ""),
                inputs=audio_input,
                outputs=[transcript_box, t_status],
            )

        # ── Tab 2: Voice Chat ────────────────────────────────────────────
        with gr.Tab("Voice Chat"):
            gr.Markdown(
                "Speak in English or Hindi — the assistant replies in kind and reads the answer aloud."
            )

            history_state = gr.State([])  # list[dict] — OpenAI message format

            chatbot = gr.Chatbot(label="Conversation", height=420, bubble_full_width=False)

            with gr.Row():
                mic_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Hold to record",
                )
                reply_audio = gr.Audio(
                    label="Assistant reply",
                    autoplay=True,
                    interactive=False,
                )

            with gr.Row():
                send_btn = gr.Button("Send", variant="primary", scale=3)
                clear_btn = gr.Button("Clear chat", scale=1)
                c_status = gr.Textbox(label="Status", interactive=False, max_lines=1, scale=2)

            send_btn.click(
                fn=chat_turn,
                inputs=[mic_input, history_state, chatbot],
                outputs=[chatbot, history_state, reply_audio, c_status],
            )
            clear_btn.click(
                fn=clear_chat,
                outputs=[chatbot, history_state, reply_audio, c_status],
            )


if __name__ == "__main__":
    demo.launch()
