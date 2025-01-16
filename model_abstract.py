import wave
import contextlib
import locale
import whisper
import logging
import librosa
import torch
import torchaudio
import torchaudio.functional as F
import numpy

import json
import time

from nemo.collections.asr.models import EncDecCTCModel, EncDecRNNTBPEModel
from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessor as NeMoAudioToMelSpectrogramPreprocessor
)

from nemo.collections.asr.modules.audio_preprocessing import AudioToMelSpectrogramPreprocessor

from nemo.collections.asr.parts.preprocessing.features import (
    FilterbankFeaturesTA as NeMoFilterbankFeaturesTA
)

import nemo.collections.asr.modules

from vosk import Model, KaldiRecognizer, GpuInit
from torch import Tensor
from torchaudio.functional._alignment import TokenSpan
from pyannote.audio import Pipeline
from utils import segment_audio, RUSSIAN_VOCABULARY
from conf.config import HF_TOKEN 


file_log = logging.FileHandler('asr_logs.log')
console_out = logging.StreamHandler()

logging.basicConfig(handlers=(file_log, console_out), 
                    format='[%(asctime)s | %(levelname)s]: %(message)s', 
                    datefmt='%m.%d.%Y %H:%M:%S',
                    level=logging.INFO)

locale.getpreferredencoding = lambda: "UTF-8"
DEVICE_DEFAULT = 'cpu'
BATCH_SIZE_DEFAULT = 4
DURATION_TO_SPLIT_DEFAULT = 30
NUM_WORKERS_DEFAULT = 6


class ModelAbstract:
    """Класс для обработки wav-аудио файла и нахождения проблемных
    мест в нём. Класс является приватным интсрументарием, для полноценного
    использования необходимо использовать класс ASRInference."""

    def __init__(self, model_name: str, model_config: dict):
        self._model_name = model_name
        self._model_config = model_config
        self._emission = None

        logging.info('Класс ModelAbstract был запущен')
        
        self._load_config()
        logging.info('Конфиг загружен и инициализирован')
        self._load_model()
        logging.info('Модель загружена')


    def _load_model(self):
        """Метод для загрузки ASR-модели в зависимости от выбранной модели
        и её конфигурации."""

        if self._model_name == 'GigaAM' and self._model_config['model_type'] == 'CTC':
            self._model = EncDecCTCModel.from_config_file(self._model_config['model_config'])
            self._ckpt = torch.load(self._model_config['model_weights'], map_location=self._device)
            self._model.load_state_dict(self._ckpt, strict=False)
            self._model.eval()
            self._model = self._model.to(self._device)

        elif self._model_name == 'GigaAM' and self._model_config['model_type'] == 'RNNT':
            self._model = EncDecRNNTBPEModel.from_config_file(self._model_config['model_config'])
            self._ckpt = torch.load(self._model_config['model_weights'], map_location=self._device)
            self._model.load_state_dict(self._ckpt, strict=False)
            self._model.eval()
            self._model = self._model.to(self._device)

        elif self._model_name == 'Whisper':
            self._model = whisper.load_model(self._model_config['model_type'],
                                             device=self._device)
        
        elif self._model_name == 'Vosk':
            if self._device == 'cuda':
                GpuInit()
            
            self._model = Model(model_path=self._model_config['model_config'])

        else:
            raise NameError('No such model was found!')
        
        
    def _load_config(self):
        """Метод для загрузки и инициализации конфигурационных
        переменных модели."""        
        
        if 'device' in self._model_config.keys():
            self._device = self._model_config['device']
        else:
            self._device = DEVICE_DEFAULT
        
        if 'batch_size' in self._model_config.keys():
            self._batch_size = self._model_config['batch_size']
        else:
            self._batch_size = BATCH_SIZE_DEFAULT 
            
        if 'num_workers' in self._model_config.keys():
            self._num_workers = self._model_config['num_workers']
        else:
            self._num_workers = NUM_WORKERS_DEFAULT 
            
        if 'duration_to_split' in self._model_config.keys():
            self._duration_to_split = self._model_config['duration_to_split']
        else:
            self._duration_to_split = DURATION_TO_SPLIT_DEFAULT
        

    def _voice_activity_detection(self, waveform: str) -> list[numpy.ndarray]:
        """Метод для разбиения wav-аудиофайла на сегменты с помощью
        HuggingFace модели pyannotate."""
        
        pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection", use_auth_token=HF_TOKEN
        )
        pipeline = pipeline.to(torch.device(self._device))
        torch.cuda.empty_cache()
        segments, self._boundaries = segment_audio(waveform, pipeline)
        
        return segments
    

    def _get_alignments(self, emission: Tensor, tokens: list[int]) -> tuple[Tensor]:
        """Метод для получения тензоров выравнивания и скоров на основе переданных
        эмиссий и списков токенов для каждого сегмента либо для всего wav файла."""
        
        targets = torch.tensor([tokens], dtype=torch.int32)
        alignments, scores = F.forced_align(emission, targets, blank=0)

        alignments, scores = alignments[0], scores[0]
        scores = scores.exp()
    

        return alignments, scores


    def _unflatten(self, list_: list[TokenSpan], 
                   lengths: list[int]) -> list[list[TokenSpan]]:
        """Разглаживание TokenSpan объектов. На выходе выдаёт
        выравненный список TokenSpan."""
        
        assert len(list_) == sum(lengths)
        i = 0
        ret = []

        for l in lengths:
            ret.append(list_[i: i + l])
            i += l
            
        return ret


    def _format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        full_seconds = int(seconds)
        milliseconds = int((seconds - full_seconds) * 100)

        if hours > 0:
            return f"{hours:02}:{minutes:02}:{full_seconds:02}:{milliseconds:02}"
        else:
            return f"{minutes:02}:{full_seconds:02}:{milliseconds:02}"


    def _get_wav_duration(self, speech_filename: str) -> int:
        """Метод необходим для получения длительности wav файла 
        в секундах."""
        
        with contextlib.closing(wave.open(speech_filename, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)

            return int(duration)


    def _tokenize(self, audio_segments: str) -> list[list[str]]:
        """Метод для токенизации аудио-сегмента или всего wav файла."""

        if self._model_name == 'GigaAM':
            self._transcribe_result = self._model.transcribe(audio_segments,
                                                             batch_size=self._batch_size,
                                                             return_hypotheses=True,
                                                             num_workers=self._num_workers)
            self._transcribe_result = self._transcribe_result if self._model_config['model_type'] == 'CTC' else self._transcribe_result[0]
            self._emission = []

            for trascribe_res in self._transcribe_result:
                emiss = trascribe_res.alignments
                emiss = emiss.unsqueeze(0)

                self._emission.append(emiss)

            return [transcribe_res.text.split(' ') for transcribe_res in self._transcribe_result]


    def _score(self, spans: list) -> int:
        """Метод для получения скоров по списку spans объектов."""
        
        return sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)


    def _split_wav(self, speech_filename: str,
                   boundaries: list[list[float]]) -> list[tuple[Tensor]]:
        """Метод для разбиения файла на waveform сегменты по 
        boundaries (временные метрики сегментов). Возвращает список
        кортежей из waveform объектов (тензоров)."""
        
        waveform, sample_rate = torchaudio.load(speech_filename)
        split_segments = []

        for start_time, end_time in boundaries:
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)

            segment = waveform[:, start_sample:end_sample]
            split_segments.append((segment, sample_rate))

        return split_segments
    
    
    def _preview_word(self, waveform: Tensor, spans: list, num_frames: int, transcript: str,
                      segment_addition: float, sample_rate: int = 16000) -> list:
        """Метод для конвертирования токена по сегменту либо
        полной аудиозаписи в список из токена и его временных меток."""
        
        ratio = waveform.size(1) / num_frames
        x0 = int(ratio * spans[0].start)
        x1 = int(ratio * spans[-1].end)
        segment = waveform[:, x0:x1]
        
        alignment_result = [transcript, (x0 / sample_rate) + segment_addition,
                (x1 / sample_rate) + segment_addition]

        return alignment_result
    
    
    def _align(self, emiss: Tensor, transc: list[str],
               waveform: Tensor) -> list[list]:
        """Метод используется для получения выравнивания по эмиссии 
        и токенам сегмента или полной аудиозаписи. Возвращает список списков
        токенов и их временных меток."""
        
        tokenized_transcript = [self._vocabulary[c] for word in transc for c in word]
        aligned_tokens, alignment_scores = self._get_alignments(emiss, tokenized_transcript)
    
        token_spans = F.merge_tokens(aligned_tokens, alignment_scores)
        word_spans = self._unflatten(token_spans, [len(word) for word in transc])
        num_frames = emiss.size(1)

        forced_align = []
        segment_addition = self._boundaries[0][0] if self._duration_to_split else 0

        for i in range(0, len(transc)):
            if len(word_spans[i]):
                forced_align.append(self._preview_word(waveform, word_spans[i], num_frames, 
                                                   transc[i], segment_addition))
            
            
        if self._duration_to_split:
            if len(self._boundaries) > 1:
                self._boundaries.pop(0)

        return forced_align


    def transcribe(self, speech_filename: str) -> list[list[str]]:
        """Метод для получения транскрибации из аудиофайла (возвращает
        список токенов | список списков токенов)."""
        
        logging.info('Начало траскрибация аудиозаписи')

        if self._model_name == 'GigaAM':
            self._duration_to_split = self._get_wav_duration(speech_filename) > 35 # todo: DURATION_TO_SPLIT переменную сделать
            
            # todo: адаптировать под короткие записи с сегментацией в один элемент
            if self._duration_to_split:
                self._segments = self._voice_activity_detection(speech_filename)
                logging.info(f'Аудиозапись разбита на {len(self._segments)} сегментов')
            else:
                speech_audio, sr = librosa.load(speech_filename)
                self._segments = [numpy.array(speech_audio)]

            self._wav_transcript = self._tokenize(self._segments)

            logging.info('Транскрибация аудиофайла завершена')

        elif self._model_name == 'Whisper':
            self._wav_transcribe = self._model.transcribe(speech_filename, word_timestamps=True)
            self._wav_transcript = []

            for segment in self._wav_transcribe['segments']:
                for word in segment['words']:
                    token = word['word'].strip()

                    self._wav_transcript.append(token)
            
            logging.info('Транскрибация аудиофайла завершена')
            self._wav_transcript = [self._wav_transcript]
        
        elif self._model_name == 'Vosk':
            wf = wave.open(speech_filename, 'rb')

            recognizer = KaldiRecognizer(self._model, wf.getframerate())
            recognizer.SetWords(True)
            recognizer.SetPartialWords(True)
        
            self._wav_results = []
            self._wav_results_magic = []

            st = time.time()
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if recognizer.AcceptWaveform(data):
                    self._wav_results.append(recognizer.Result())
                else:
                    self._wav_results_magic = 'ERROR'
         
            en = time.time()
            self._wav_transcript = []
            
            for transcript in self._wav_results:
                converted = json.loads(transcript)
                
                try:
                    for res in converted['result']:
                        self._wav_transcript.append(res['word'])

                except KeyError:
                    pass

            self._wav_transcript = [self._wav_transcript]
            
            logging.info('Транскрибация аудиофайла завершена')
        
        else:
            raise NameError('No such model was found!')
        
        transcript_result = []

        for token_segment in self._wav_transcript:
            transcript_result.extend(token_segment)

        return transcript_result
        

    def forced_align(self, speech_filename: str) -> list[list]:
        """Метод для выравнивания и получения временного распределения 
        по алфавиту токенов на спектрограмме. Возвращает список списков, в каждом
        подсписке находится 3 элемента: токен, время начала произношения токена,
        время конца."""

        # todo: переписать без использования ошибки             
        try:
          self._emission.bool
        except Exception as e:
          self.transcribe(speech_filename)
          
        logging.info('Выравнивание аудиозаписи началось')

        if self._model_name == 'GigaAM':
            self._vocabulary = RUSSIAN_VOCABULARY
            self._vocabulary[' '] = 33
            
            forced_alignments = []

            if self._duration_to_split:
                self._waveform_segments = self._split_wav(speech_filename, self._boundaries)
            else:
                segment, sample_rate = torchaudio.load(speech_filename)
                self._waveform_segments = [(segment, sample_rate)]
            
            logging.info('Аудиозапись разбита на сегменты для выравнивания')
            
            for emiss, transc, (waveform_seg, sample_rate) in zip(self._emission, self._wav_transcript,
                                                self._waveform_segments):
                forced_alignments.extend(self._align(emiss, transc, waveform_seg))
                
            logging.info('Выравнивание завершено')

            return forced_alignments

        elif self._model_name == 'Whisper':
            forced_alignments = []

            for segment in self._wav_transcribe['segments']:
                for word in segment['words']:
                    token = word['word'].strip()
                    ts_begin = word['start']
                    ts_end = word['end']

                    forced_alignments.append([token, ts_begin, ts_end])
                
            logging.info('Выравнивание завершено')

            return forced_alignments
        
        elif self._model_name == 'Vosk':
            forced_alignments = []
            
            for transcript in self._wav_results:
                converted = json.loads(transcript)
                
                try:
                    for i in converted['result']:
                        forced_alignments.append([i['word'], i['start'], i['end']])

                except KeyError:
                    pass
                
            logging.info('Выравнивание завершено')
                
            return forced_alignments
        
        else:
            raise NameError('No such model was found!')
