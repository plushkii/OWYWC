import json
import os
import numpy as np
import subprocess
import torchaudio

from io import BytesIO
from typing import List, Tuple

from pydub import AudioSegment
from pyannote.audio import Pipeline

from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessor as NeMoAudioToMelSpectrogramPreprocessor,
)
from nemo.collections.asr.parts.preprocessing.features import (
    FilterbankFeaturesTA as NeMoFilterbankFeaturesTA,
)


RUSSIAN_VOCABULARY = {letter: index + 1 for index, letter in enumerate("абвгдежзийклмнопрстуфхцчшщъыьэюя")}


class FilterbankFeaturesTA(NeMoFilterbankFeaturesTA):
    def __init__(self, mel_scale: str = "htk", wkwargs=None, **kwargs):
        if "window_size" in kwargs:
            del kwargs["window_size"]
        if "window_stride" in kwargs:
            del kwargs["window_stride"]

        super().__init__(**kwargs)

        self._mel_spec_extractor: torchaudio.transforms.MelSpectrogram = (
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self._sample_rate,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels=kwargs["nfilt"],
                window_fn=self.torch_windows[kwargs["window"]],
                mel_scale=mel_scale,
                norm=kwargs["mel_norm"],
                n_fft=kwargs["n_fft"],
                f_max=kwargs.get("highfreq", None),
                f_min=kwargs.get("lowfreq", 0),
                wkwargs=wkwargs,
            )
        )


class AudioToMelSpectrogramPreprocessor(NeMoAudioToMelSpectrogramPreprocessor):
    def __init__(self, mel_scale: str = "htk", **kwargs):
        super().__init__(**kwargs)
        kwargs["nfilt"] = kwargs["features"]
        del kwargs["features"]
        self.featurizer = (
            FilterbankFeaturesTA(  # Deprecated arguments; kept for config compatibility
                mel_scale=mel_scale,
                **kwargs,
            )
        )


def audiosegment_to_numpy(audiosegment: AudioSegment) -> np.ndarray:
    """Convert AudioSegment to numpy array."""
    samples = np.array(audiosegment.get_array_of_samples())
    if audiosegment.channels == 2:
        samples = samples.reshape((-1, 2))

    samples = samples.astype(np.float32, order="C") / 32768.0
    return samples


def segment_audio(
    audio_path: str,
    pipeline: Pipeline,
    max_duration: float = 22.0,
    min_duration: float = 15.0,
    new_chunk_threshold: float = 0.2,
) -> Tuple[List[np.ndarray], List[List[float]]]:

    audio = AudioSegment.from_wav(audio_path)
    audio_bytes = BytesIO()
    audio.export(audio_bytes, format="wav")
    audio_bytes.seek(0)

    sad_segments = pipeline({"uri": "filename", "audio": audio_bytes})

    segments = []
    curr_duration = 0
    curr_start = 0
    curr_end = 0
    boundaries = []
    
    for segment in sad_segments.get_timeline().support():
        start = max(0, segment.start)
        end = min(len(audio) / 1000, segment.end)
        if (
            curr_duration > min_duration and start - curr_end > new_chunk_threshold
        ) or (curr_duration + (end - curr_end) > max_duration):
            audio_segment = audiosegment_to_numpy(
                audio[curr_start * 1000 : curr_end * 1000]
            )
            segments.append(audio_segment)
            boundaries.append([curr_start, curr_end])
            curr_start = start

        curr_end = end
        curr_duration = curr_end - curr_start

    if curr_duration != 0:
        audio_segment = audiosegment_to_numpy(
            audio[curr_start * 1000 : curr_end * 1000]
        )
        segments.append(audio_segment)
        boundaries.append([curr_start, curr_end])

    return segments, boundaries


def convert_wav(speech_filename):
    new_filename = f'{os.path.basename(speech_filename).split(".")[0]}_patched.wav'
    command = ['ffmpeg', '-i', speech_filename, '-ac', '1', '-ar', '16000', new_filename, '-y']

    subprocess.run(command)
    
    return new_filename


# Функция "Нормализации" transcribe, для сравнения с эталоном при тестировании

def normalize_transcribe(transcribe):
    normalized_transcribe = []
    sym_list = ('.', ',', '-', '?', '!', ';', ':', '"', "'", '(', ')', '—')
    for token in transcribe:
        tk = token.lower()
        for sym in sym_list:
            tk = tk.replace(sym, '').strip()
    normalized_transcribe.append(tk)
    return normalized_transcribe

# Функция "Нормализации" forced_alignment, для сравнения с эталоном при тестировании

def normalize_forced_alignment(alignment):
    normalized_alignment = alignment
    for i in range(len(alignment)):
        normalized_alignment[i] = [alignment[i][0], round(alignment[i][1], 2), round(alignment[i][2], 2)]
    return normalized_alignment
