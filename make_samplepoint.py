#!/usr/bin/env python3
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def to_mono(data: np.ndarray, mode: str) -> np.ndarray:
    """
    Convert input audio to mono float32.

    data shape:
      mono:   (samples,)
      stereo: (samples, channels)

    mode:
      left  -> use left channel only
      right -> use right channel only
      mix   -> average all channels
    """
    if data.ndim == 1:
        return data.astype(np.float32)

    if mode == "left":
        return data[:, 0].astype(np.float32)

    if mode == "right":
        if data.shape[1] < 2:
            raise ValueError(
                "Right channel was requested, but the file has fewer than 2 channels."
            )
        return data[:, 1].astype(np.float32)

    if mode == "mix":
        return np.mean(data, axis=1).astype(np.float32)

    raise ValueError(f"Unsupported channel mode: {mode}")


def normalized_cross_correlation_search(
    x: np.ndarray,
    template_start: int,
    window: int,
    search_start: int,
    search_end: int,
) -> tuple[int, float]:
    """
    Search for a segment similar to x[template_start:template_start + window].

    Returns:
      best_position, best_score

    best_score:
      approximately -1.0 to 1.0.
      Higher means more similar.
    """
    n = len(x)

    template_end = template_start + window
    if template_start < 0 or template_end > n:
        raise ValueError("Template range is outside the audio data.")

    search_start = max(search_start, 0)
    search_end = min(search_end, n - window)

    if search_start >= search_end:
        raise ValueError("Invalid search range. Try lowering --min-gap or --window.")

    template = x[template_start:template_end].astype(np.float64)
    template = template - np.mean(template)
    template_norm = np.linalg.norm(template)

    if template_norm < 1e-12:
        raise ValueError(
            "Template is almost silent or flat. Choose another --start point."
        )

    # corr[i] corresponds to dot(x[i:i+window], template)
    corr = np.correlate(x.astype(np.float64), template, mode="valid")

    # Compute local variance efficiently using cumulative sums.
    x64 = x.astype(np.float64)
    csum = np.concatenate([[0.0], np.cumsum(x64)])
    csum2 = np.concatenate([[0.0], np.cumsum(x64 * x64)])

    idx = np.arange(search_start, search_end + 1)

    sums = csum[idx + window] - csum[idx]
    sums2 = csum2[idx + window] - csum2[idx]

    means = sums / window
    variances = sums2 - 2.0 * means * sums + window * means * means
    variances = np.maximum(variances, 1e-12)

    denom = np.sqrt(variances) * template_norm
    scores = corr[idx] / denom

    best_local_index = int(np.argmax(scores))
    best_position = int(idx[best_local_index])
    best_score = float(scores[best_local_index])

    return best_position, best_score


def discontinuity_cost(x: np.ndarray, loop_start: int, loop_end: int) -> float:
    """
    Estimate click risk at the loop boundary.

    loop_end is treated as the jump point:
      playback reaches loop_end - 1, then jumps to loop_start.

    Lower is better.
    """
    n = len(x)

    if loop_start < 1:
        return float("inf")

    if loop_start + 1 >= n:
        return float("inf")

    if loop_end < 2:
        return float("inf")

    if loop_end >= n:
        return float("inf")

    value_gap = float(x[loop_end - 1] - x[loop_start])

    slope_before_end = float(x[loop_end - 1] - x[loop_end - 2])
    slope_after_start = float(x[loop_start + 1] - x[loop_start])
    slope_gap = slope_before_end - slope_after_start

    return value_gap * value_gap + 0.25 * slope_gap * slope_gap


def refine_loop_end(
    x: np.ndarray,
    loop_start: int,
    rough_loop_end: int,
    refine_radius: int,
    window: int,
    similarity_weight: float = 0.15,
) -> int:
    """
    Refine loop end around the rough candidate.

    The final score combines:
      - boundary discontinuity
      - local waveform similarity
    """
    n = len(x)

    a = max(loop_start + 8, rough_loop_end - refine_radius)
    b = min(n - window - 1, rough_loop_end + refine_radius)

    if a >= b:
        return rough_loop_end

    template = x[loop_start : loop_start + window].astype(np.float64)
    template = template - np.mean(template)
    template_norm = np.linalg.norm(template) + 1e-12

    best_pos = rough_loop_end
    best_cost = float("inf")

    for pos in range(a, b + 1):
        candidate = x[pos : pos + window].astype(np.float64)
        candidate = candidate - np.mean(candidate)
        candidate_norm = np.linalg.norm(candidate) + 1e-12

        similarity = float(
            np.dot(template, candidate) / (template_norm * candidate_norm)
        )
        similarity_cost = 1.0 - similarity

        boundary_cost = discontinuity_cost(x, loop_start, pos)

        cost = boundary_cost + similarity_weight * similarity_cost

        if cost < best_cost:
            best_cost = cost
            best_pos = pos

    return best_pos


def crossfade_loop_end(
    audio: np.ndarray,
    loop_start: int,
    loop_end: int,
    window: int,
) -> tuple[np.ndarray, int]:
    """
    Crossfade the loop end into the loop start over up to window samples.
    """
    fade_length = min(window, loop_end - loop_start, loop_end)
    if fade_length <= 1:
        raise ValueError("Not enough loop length to apply crossfade.")

    result = audio.copy()
    tail_start = loop_end - fade_length

    tail = result[tail_start:loop_end].copy()
    head = result[loop_start : loop_start + fade_length].copy()
    fade = np.linspace(0.0, 1.0, fade_length, dtype=np.float32)

    result[tail_start:loop_end] = tail * (1.0 - fade) + head * fade
    return result, fade_length


def resample_linear(
    audio: np.ndarray,
    source_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    """
    Resample mono audio using linear interpolation.
    """
    if source_sample_rate <= 0:
        raise ValueError("Input sample rate must be > 0.")

    if source_sample_rate == target_sample_rate:
        return audio.astype(np.float32)

    if len(audio) < 2:
        raise ValueError("Audio is too short to resample.")

    target_length = int(round(len(audio) * target_sample_rate / source_sample_rate))
    if target_length < 2:
        raise ValueError("Resampled audio would be too short.")

    source_positions = np.arange(len(audio), dtype=np.float64)
    target_positions = (
        np.arange(target_length, dtype=np.float64)
        * source_sample_rate
        / target_sample_rate
    )
    target_positions = np.minimum(target_positions, len(audio) - 1)

    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def quantize_to_8bit_float(audio: np.ndarray) -> np.ndarray:
    """
    Quantize float audio to unsigned 8-bit PCM levels, then return float audio.
    """
    clipped = np.clip(audio, -1.0, 1.0)
    pcm8 = np.rint((clipped + 1.0) * 127.5).astype(np.uint8)
    return (pcm8.astype(np.float32) / 127.5 - 1.0).astype(np.float32)


def quantize_to_16bit_float(audio: np.ndarray) -> np.ndarray:
    """
    Quantize float audio to signed 16-bit PCM levels, then return float audio.
    """
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = np.rint(clipped * 32767.0).astype(np.int16)
    return (pcm16.astype(np.float32) / 32767.0).astype(np.float32)


def wav_subtype_bits(subtype: str) -> int | None:
    """
    Return a rough source bit depth for common SoundFile WAV subtypes.
    """
    subtype_bits = {
        "PCM_U8": 8,
        "PCM_16": 16,
        "PCM_24": 24,
        "PCM_32": 32,
        "FLOAT": 32,
        "DOUBLE": 64,
        "ULAW": 8,
        "ALAW": 8,
    }
    return subtype_bits.get(subtype)


def prepare_default_audio(
    audio: np.ndarray,
    sample_rate: int,
    source_bits_per_sample: int | None,
) -> tuple[np.ndarray, int, int]:
    """
    Keep default processing at or below 44100 Hz / 16-bit.
    """
    target_sample_rate = 44100 if sample_rate > 44100 else sample_rate

    if sample_rate > target_sample_rate:
        audio = resample_linear(audio, sample_rate, target_sample_rate)

    if source_bits_per_sample is None or source_bits_per_sample > 16:
        audio = quantize_to_16bit_float(audio)

    return audio, target_sample_rate, 16


def prepare_22050_8bit_audio(
    audio: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, int, int]:
    """
    Convert audio to 22050 Hz and 8-bit-equivalent samples.
    """
    target_sample_rate = 22050
    audio = resample_linear(audio, sample_rate, target_sample_rate)
    audio = quantize_to_8bit_float(audio)
    return audio, target_sample_rate, 8


def float_to_pcm16_bytes(x: np.ndarray) -> bytes:
    """
    Convert float audio in roughly [-1.0, 1.0] to little-endian PCM16.
    """
    y = np.clip(x, -1.0, 1.0)
    y = (y * 32767.0).astype("<i2")
    return y.tobytes()


def float_to_pcm8_bytes(x: np.ndarray) -> bytes:
    """
    Convert float audio in roughly [-1.0, 1.0] to unsigned PCM8.
    """
    y = np.clip(x, -1.0, 1.0)
    y = np.rint((y + 1.0) * 127.5).astype(np.uint8)
    return y.tobytes()


def float_to_pcm_bytes(x: np.ndarray, bits_per_sample: int) -> bytes:
    if bits_per_sample == 8:
        return float_to_pcm8_bytes(x)

    if bits_per_sample == 16:
        return float_to_pcm16_bytes(x)

    raise ValueError("bits_per_sample must be 8 or 16.")


def write_wav_with_smpl_loop(
    path: Path,
    audio: np.ndarray,
    sample_rate: int,
    bits_per_sample: int,
    loop_start: int,
    loop_end_exclusive: int,
) -> None:
    """
    Write mono PCM WAV with an smpl chunk.

    Internally:
      loop_end_exclusive is the first sample after the loop.

    WAV smpl loop end is commonly stored as inclusive,
    so this writes loop_end_exclusive - 1.
    """
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim != 1:
        raise ValueError("This writer expects mono audio.")

    if loop_start < 0:
        raise ValueError("loop_start must be >= 0.")

    if loop_end_exclusive <= loop_start:
        raise ValueError("loop_end must be greater than loop_start.")

    if loop_end_exclusive > len(audio):
        raise ValueError("loop_end is outside the cropped audio.")

    num_channels = 1
    block_align = num_channels * bits_per_sample // 8
    byte_rate = sample_rate * block_align

    pcm_data = float_to_pcm_bytes(audio, bits_per_sample)

    fmt_chunk_data = struct.pack(
        "<HHIIHH",
        1,  # Audio format: PCM
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    )

    # smpl chunk
    sample_period_ns = int(round(1_000_000_000 / sample_rate))
    midi_unity_note = 60
    midi_pitch_fraction = 0
    smpte_format = 0
    smpte_offset = 0
    num_sample_loops = 1
    sampler_data = 0

    smpl_header = struct.pack(
        "<9I",
        0,  # manufacturer
        0,  # product
        sample_period_ns,
        midi_unity_note,
        midi_pitch_fraction,
        smpte_format,
        smpte_offset,
        num_sample_loops,
        sampler_data,
    )

    loop_cue_point_id = 0
    loop_type_forward = 0
    smpl_loop_start = int(loop_start)
    smpl_loop_end = int(loop_end_exclusive - 1)
    loop_fraction = 0
    loop_play_count = 0

    smpl_loop = struct.pack(
        "<6I",
        loop_cue_point_id,
        loop_type_forward,
        smpl_loop_start,
        smpl_loop_end,
        loop_fraction,
        loop_play_count,
    )

    smpl_chunk_data = smpl_header + smpl_loop

    chunks: list[bytes] = []

    def add_chunk(chunk_id: bytes, chunk_data: bytes) -> None:
        chunks.append(chunk_id)
        chunks.append(struct.pack("<I", len(chunk_data)))
        chunks.append(chunk_data)

        # WAV chunks must be word-aligned.
        if len(chunk_data) % 2 == 1:
            chunks.append(b"\x00")

    add_chunk(b"fmt ", fmt_chunk_data)
    add_chunk(b"smpl", smpl_chunk_data)
    add_chunk(b"data", pcm_data)

    body = b"".join(chunks)
    riff_size = 4 + len(body)

    with path.open("wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", riff_size))
        f.write(b"WAVE")
        f.write(body)


def validate_args(
    audio: np.ndarray,
    start: int,
    window: int,
    min_gap: int,
    refine: int,
    crop_margin: int,
    search_end: int | None,
) -> None:
    n = len(audio)

    if start < 0:
        raise ValueError("--start must be >= 0.")

    if window <= 0:
        raise ValueError("--window must be > 0.")

    if min_gap <= 0:
        raise ValueError("--min-gap must be > 0.")

    if refine < 0:
        raise ValueError("--refine must be >= 0.")

    if crop_margin < 0:
        raise ValueError("--crop-margin must be >= 0.")

    if start + window >= n:
        raise ValueError(
            f"--start + --window is outside the audio data "
            f"({start} + {window} >= {n}). "
            "Move --start earlier or lower --window."
        )

    max_search_end = n - window - 1
    if search_end is not None:
        max_search_end = min(max_search_end, search_end)

    max_min_gap = max_search_end - start
    if min_gap > max_min_gap:
        raise ValueError(
            f"--min-gap is too large for this audio and --start value. "
            f"The last searchable loop-end position is {max_search_end}, "
            f"so --min-gap must be <= {max_min_gap}. "
            f"Current values require {start} + {min_gap} + {window} samples, "
            f"but the audio has {n} samples. "
            f"Try --min-gap {max(1, max_min_gap // 2)} or choose an earlier --start."
        )

    if search_end is not None:
        if search_end <= start + min_gap:
            raise ValueError(
                f"--search-end must be greater than --start + --min-gap "
                f"({search_end} <= {start} + {min_gap})."
            )

        if search_end + window >= n:
            raise ValueError(
                f"--search-end + --window is outside the audio data "
                f"({search_end} + {window} >= {n}). "
                "Lower --search-end or --window."
            )


def prompt_scan_to_end(start: int, window: int, search_end: int) -> int:
    """
    Ask whether to scan from just after the template window to the effective end.

    Returns the min_gap to use when accepted.
    """
    min_gap = window
    search_start = start + min_gap

    print(
        "--min-gap was not specified. "
        f"Scan loop-end candidates from sample {search_start} "
        f"to {search_end}? [Y/n]: ",
        end="",
        flush=True,
    )

    try:
        answer = input().strip().lower()
    except EOFError as exc:
        raise RuntimeError(
            "No answer was provided. Specify --min-gap N when running non-interactively."
        ) from exc
    if answer in ("", "y", "yes"):
        return min_gap

    if answer in ("n", "no"):
        raise RuntimeError("Canceled. Specify --min-gap N to set the search distance.")

    raise RuntimeError("Canceled. Please answer Y or n.")


def default_output_path(input_path: Path, mode_22: bool) -> Path:
    """
    Build the output path used when the output argument is omitted.
    """
    suffix = "_lp_22" if mode_22 else "_lp"
    return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


def append_score_to_path(path: Path, score: float) -> Path:
    """
    Add the similarity score to an output file name.
    """
    return path.with_name(f"{path.stem}_score{score:.4f}{path.suffix}")


def main() -> None:
    if len(sys.argv) == 1:
        print(
            """Usage:
  python make_samplepoint.py INPUT.wav [OUTPUT.wav] --start START_SAMPLE [options]

Example:
  python make_samplepoint.py input.wav --start 52920 --window 4096 --min-gap 22050
  python make_samplepoint.py input.wav output_looped.wav --start 52920 --window 4096 --min-gap 22050

Required:
  INPUT.wav              Input WAV file
  --start N              Loop start sample index

Optional:
  OUTPUT.wav             Output WAV file.
                          If omitted, writes INPUT_lp.wav or INPUT_lp_22.wav.

Common options:
  --22                  Convert to 22050 Hz / 8-bit before processing.
  --window N             Template window length in samples. Default: 4096
  --min-gap N            Minimum search distance after start.
                          If omitted, ask whether to scan to the waveform end.
  --threshold F          Similarity threshold. Default: 0.85
  --refine N             Refine radius in samples. Default: 1024
  --crop-margin N        Samples to keep after loop end. Default: 0
  --channel MODE         left, right, or mix. Default: left

Default mode converts sources above 44100 Hz / 16-bit to 44100 Hz / 16-bit
before searching and writing.

For full help:
  python make_samplepoint.py --help
"""
        )
        return
    parser = argparse.ArgumentParser(
        description=(
            "Find a loop end similar to a specified loop start, "
            "crop the WAV, and save a new mono WAV with smpl loop metadata."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input WAV file.",
    )

    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help=(
            "Output WAV file. If omitted, writes INPUT_lp.wav, "
            "or INPUT_lp_22.wav when --22 is used."
        ),
    )

    parser.add_argument(
        "--start",
        required=True,
        type=int,
        help="Loop start sample index. Example: 52920",
    )

    parser.add_argument(
        "--22",
        dest="mode_22",
        action="store_true",
        help="Convert to 22050 Hz / 8-bit before searching and writing.",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=4096,
        help="Template window length in samples. Default: 4096",
    )

    parser.add_argument(
        "--min-gap",
        type=int,
        default=None,
        help=(
            "Minimum distance after loop start before searching, in samples. "
            "If omitted, ask whether to scan to the waveform end."
        ),
    )

    parser.add_argument(
        "--search-end",
        type=int,
        default=None,
        help="Optional search end sample index. Default: near the end of the file.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help=(
            "Minimum score to save without crossfade. "
            "Lower scores are crossfaded and saved with the score in the file name. "
            "Default: 0.85"
        ),
    )

    parser.add_argument(
        "--refine",
        type=int,
        default=1024,
        help="Refine radius around found loop end, in samples. Default: 1024",
    )

    parser.add_argument(
        "--crop-margin",
        type=int,
        default=0,
        help="Samples to keep after loop end before cropping. Default: 0",
    )

    parser.add_argument(
        "--channel",
        choices=["left", "right", "mix"],
        default="left",
        help="For stereo files: left, right, or mix. Default: left",
    )

    args = parser.parse_args()
    output_path = args.output or default_output_path(args.input, args.mode_22)

    input_info = sf.info(args.input)
    source_bits_per_sample = wav_subtype_bits(input_info.subtype)

    data, source_sample_rate = sf.read(args.input, always_2d=False)
    audio = to_mono(data, args.channel)
    source_samples = len(audio)

    sample_rate = source_sample_rate
    bits_per_sample = 16

    if args.mode_22:
        audio, sample_rate, bits_per_sample = prepare_22050_8bit_audio(
            audio=audio,
            sample_rate=sample_rate,
        )
    else:
        audio, sample_rate, bits_per_sample = prepare_default_audio(
            audio=audio,
            sample_rate=sample_rate,
            source_bits_per_sample=source_bits_per_sample,
        )

    if args.search_end is None:
        search_end = len(audio) - args.window - 1
    else:
        search_end = args.search_end

    if args.min_gap is None:
        args.min_gap = prompt_scan_to_end(
            start=args.start,
            window=args.window,
            search_end=search_end,
        )

    validate_args(
        audio=audio,
        start=args.start,
        window=args.window,
        min_gap=args.min_gap,
        refine=args.refine,
        crop_margin=args.crop_margin,
        search_end=args.search_end,
    )

    loop_start = args.start
    search_start = loop_start + args.min_gap

    rough_loop_end, score = normalized_cross_correlation_search(
        x=audio,
        template_start=loop_start,
        window=args.window,
        search_start=search_start,
        search_end=search_end,
    )

    use_crossfade = score < args.threshold
    if use_crossfade:
        output_path = append_score_to_path(output_path, score)

    loop_end = refine_loop_end(
        x=audio,
        loop_start=loop_start,
        rough_loop_end=rough_loop_end,
        refine_radius=args.refine,
        window=args.window,
    )

    crop_end = min(len(audio), loop_end + args.crop_margin)
    cropped = audio[:crop_end]
    crossfade_length = 0

    if use_crossfade:
        cropped, crossfade_length = crossfade_loop_end(
            audio=cropped,
            loop_start=loop_start,
            loop_end=loop_end,
            window=args.window,
        )

    write_wav_with_smpl_loop(
        path=output_path,
        audio=cropped,
        sample_rate=sample_rate,
        bits_per_sample=bits_per_sample,
        loop_start=loop_start,
        loop_end_exclusive=loop_end,
    )

    print("Saved:", output_path)
    print("Source sample rate:", source_sample_rate)
    print("Source bit depth:", source_bits_per_sample or "unknown")
    print("Source samples:", source_samples)
    print("Sample rate:", sample_rate)
    print("Bit depth:", bits_per_sample)
    print("Input samples:", len(audio))
    print("Output samples:", len(cropped))
    print("Channel mode:", args.channel)
    print("Loop start:", loop_start)
    print("Rough loop end:", rough_loop_end)
    print("Final loop end:", loop_end)
    print("Loop length:", loop_end - loop_start)
    print("Rough similarity score:", f"{score:.4f}")
    print("Threshold:", f"{args.threshold:.4f}")
    print("Crossfade applied:", "yes" if use_crossfade else "no")
    if use_crossfade:
        print("Crossfade length:", crossfade_length)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
