# Установка необходимых библиотек 
import logging
import torch
import torchaudio
import torchaudio.functional as F
import numpy

from langchain.prompts import load_prompt
from langchain.chains.summarize import load_summarize_chain
from langchain_community.chat_models import GigaChat
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from conf.config import AUTHORIZATION_TOKEN_GIGA

from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessor as NeMoAudioToMelSpectrogramPreprocessor,
)
from nemo.collections.asr.parts.preprocessing.features import (
    FilterbankFeaturesTA as NeMoFilterbankFeaturesTA,
)

from pyannote.audio import Pipeline
from utils import segment_audio, RUSSIAN_VOCABULARY, convert_wav
from conf.config import HF_TOKEN 


class SummaryTools:
    def __init__(self, speech_filename: str, parttime: float = 180.0):
        self.file_name = speech_filename
        self.parttime = parttime

    
    def _load_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = EncDecCTCModel.from_config_file("./ctc_model_config.yaml")
        ckpt = torch.load("./ctc_model_weights.ckpt", map_location="cpu")
        self.model.load_state_dict(ckpt, strict=False)
        self.model.eval()
        self.model = self.model.to(device)
        
        # Initialize pyannote pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection", use_auth_token=HF_TOKEN
        )
        pipeline = pipeline.to(torch.device(device))
    
    
    def _format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        full_seconds = int(seconds)
        milliseconds = int((seconds - full_seconds) * 100)

        if hours > 0:
            return f"{hours:02}:{minutes:02}:{full_seconds:02}"
        else:
            return f"{minutes:02}:{full_seconds:02}"
    
        
    def transcribed_part(self, pipeline):
        """Разбивает материал на части по parttime секунд для пересказа материала по таймкодам"""
        
        speech_file = convert_wav(self.file_name)
        # Segment audio
        segments, boundaries = segment_audio(speech_file, pipeline, max_duration=self.parttime, min_duration=self.parttime - 10.0)
        # Transcribing segments
        BATCH_SIZE = 1
        transcriptions = self.model.transcribe(segments, batch_size=BATCH_SIZE)
        
        self.transcribed_parts = []
        
        for transcription, boundary in zip(transcriptions, boundaries):
            boundary_0 = self._format_time(boundary[0])
            boundary_1 = self._format_time(boundary[1])
            self.transcribed_parts.append(boundary_0, boundary_1, transcription)
            print(f"[{boundary_0} - {boundary_1}]: {transcription}\n")
        
    
    def retelling_in_part(self):
        
        self.line_for_propmt = []
        
        for start, end, transcription in self.transcribed_parts:
            self.line_for_propmt.append(f"[{start} - {end}]: {transcription}")
            
        giga = GigaChat(credentials=AUTHORIZATION_TOKEN_GIGA, model="GigaChat", verify_ssl_certs=False, profanity_check=False)

        # Укажите полный путь до файла (зависит от окружения)
        # loader = TextLoader(self.text_for_propmt)
        # documents = loader.load()

        material_map_prompt = load_prompt("summarize/summarize_materials.map.yaml")
        chain = load_summarize_chain(giga, chain_type="map_reduce", 
                                    map_prompt=material_map_prompt,
                                    #  combine_prompt=material_combine_prompt,
                                    verbose=False)

        self.retelling_materials = chain.invoke({"input_variables": "\n".join(self.line_for_propmt)})
        # res = chain.invoke(
        #         {"input_documents": documents, "map_size": map_size, "reduce_size": reduce_size}
        #     )

        print(self.retelling_materials["output_text"].replace(". ", ".\n"))
        
        return self.retelling_materials["output_text"].replace(". ", ".\n")
    
    
    def transcribed_for_clips(self, pipeline):
        """Разбивает материал на клипы по таймкодам"""
        
        speech_file = convert_wav(self.file_name)
        # Segment audio
        segments, boundaries = segment_audio(speech_file, pipeline, max_duration=30.0, min_duration=18.0)
        # Transcribing segments
        BATCH_SIZE = 10
        transcriptions = self.model.transcribe(segments, batch_size=BATCH_SIZE)
        
        self.transcribed_for_summ_clips = []
        
        for transcription, boundary in zip(transcriptions, boundaries):
            boundary_0 = self._format_time(boundary[0])
            boundary_1 = self._format_time(boundary[1])
            self.transcribed_parts.append(boundary_0, boundary_1, transcription)
            print(f"[{boundary_0} - {boundary_1}]: {transcription}\n")
    
    
    def summ_for_clips(self):
        
        self.line_for_clips = []
        
        for start, end, transcription in self.transcribed_parts:
            self.line_for_propmt.append(f"[{start} - {end}]: {transcription}")
            
        giga = GigaChat(credentials=AUTHORIZATION_TOKEN_GIGA, model="GigaChat", verify_ssl_certs=False, profanity_check=False)

        # Укажите полный путь до файла (зависит от окружения)
        # loader = TextLoader(self.text_for_propmt)
        # documents = loader.load()
        clip_map_prompt = load_prompt("summarize/summarize_for_clips.map.yaml")
        chain = load_summarize_chain(giga, chain_type="map_reduce", 
                                    map_prompt=clip_map_prompt,
                                    #  combine_prompt=material_combine_prompt,
                                    verbose=False)

        self.summ_clips = chain.invoke({"input_variables": "\n".join(self.line_for_clips)})
        # res = chain.invoke(
        #         {"input_documents": documents, "map_size": map_size, "reduce_size": reduce_size}
        #     )

        print(self.summ_clips["output_text"].replace(". ", ".\n"))
        
        return self.summ_clips["output_text"].replace(". ", ".\n")
        

        
    
    def _voice_activity_detection(self, waveform: str) -> list[numpy.ndarray]:
        """Метод для разбиения wav-аудиофайла на сегменты с помощью
        HuggingFace модели pyannotate."""
        
        pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection", use_auth_token=HF_TOKEN
        )
        pipeline = pipeline.to(torch.device(self._device))
        torch.cuda.empty_cache()
        segments, self._boundaries = segment_audio(waveform, pipeline, max_duration=180.0, min_duration=170.0)
        
        return segments
    