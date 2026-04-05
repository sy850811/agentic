import os
import tempfile

CHUNK_SIZE_LIMIT = 24 * 1024 * 1024  # 24 MB


def needs_chunking(audio_path: str) -> bool:
    return os.path.getsize(audio_path) > CHUNK_SIZE_LIMIT


def chunk_audio(audio_path: str) -> list[str]:
    """Split audio into <24MB chunks. Returns list of temp file paths."""
    from pydub import AudioSegment
    from pydub.silence import split_on_silence

    audio = AudioSegment.from_file(audio_path)
    tmp_dir = tempfile.mkdtemp(prefix="transcribe_chunks_")

    # Try silence-based splitting first
    chunks = split_on_silence(
        audio,
        min_silence_len=800,
        silence_thresh=-40,
        keep_silence=300,
    )

    # If silence splitting gives too-large chunks, hard-split at 10 minutes
    ten_min_ms = 10 * 60 * 1000
    final_chunks: list[AudioSegment] = []
    for chunk in chunks:
        if len(chunk) > ten_min_ms:
            for start in range(0, len(chunk), ten_min_ms):
                final_chunks.append(chunk[start : start + ten_min_ms])
        else:
            final_chunks.append(chunk)

    paths = []
    ext = os.path.splitext(audio_path)[1] or ".mp3"
    fmt = ext.lstrip(".")
    if fmt == "m4a":
        fmt = "mp4"  # pydub export format for m4a

    for i, chunk in enumerate(final_chunks):
        out_path = os.path.join(tmp_dir, f"chunk_{i:04d}{ext}")
        chunk.export(out_path, format=fmt)
        paths.append(out_path)

    return paths


def cleanup_chunks(paths: list[str]) -> None:
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass
    if paths:
        try:
            os.rmdir(os.path.dirname(paths[0]))
        except OSError:
            pass
