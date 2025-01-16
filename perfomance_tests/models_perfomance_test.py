import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
from random import sample
from filter_abstract import FilterAbstract

from asr_inference import ASRInference
from utils import convert_wav #, real_time_factor, get_wer, get_duration


# TODO: Добавить данные функции в глобальный utils.py
def real_time_factor(processingTime, audioLength):
    return processingTime / audioLength

def get_wer(predictions, references):
    wer = load("wer")
    try:
        wer_score = wer.compute(predictions=predictions, references=references)
    except ZeroDivisionError:
        wer_score = float('inf')

    return wer_score

def get_duration(filename):
    return librosa.get_duration(filename=filename) 



class ModelsPerfomanceTest():
    def __init__(self, models_lst, device, top, sample_count, pipeline=False):
        self.device = device
        self.top = top
        self.sample_count = sample_count
        self.pipeline = pipeline
        self.path2clips = 'datatest/ru/clean_original_clips_0.8/'
        self.models_lst = models_lst

        self._models_init()

    def _models_init(self):
        self._loaded_models = {}

        for model_name in self.models_lst:
            # Vosk
            if model_name == 'vosk-model-small-ru-0.22':
                vosk_model_config = {
                                'model_config': 'vosk-model-small-ru-0.22',
                                'device': self.device
                               }
            
                self.vosk_model = ASRInference('Vosk', vosk_model_config)
                self._loaded_models['vosk-model-small-ru-0.22'] = self.vosk_model

            # GigaAM
            if model_name == 'GigaAM_CTC':
                GigaAM_CTC_model_config = {
                                'model_config': 'conf/ctc_model_config.yaml',
                                'model_weights': '../ml/ctc_model_weights.ckpt',
                                'model_type': 'CTC',
                                'device': self.device, 
                                'batch_size': 1
                               }
            
                self.GigaAM_CTC_model = ASRInference('GigaAM', GigaAM_CTC_model_config)
                self._loaded_models['GigaAM_CTC'] = self.GigaAM_CTC_model
                
            # Whisper
            if model_name == 'whisper_tiny':
                self.whisper_model_tiny = ASRInference('Whisper', {'device': self.device, 'model_type': 'tiny'})
                self._loaded_models['whisper_tiny'] = self.whisper_model_tiny

            if model_name == 'whisper_base':
                self.whisper_model_base = ASRInference('Whisper', {'device': self.device, 'model_type': 'base'})
                self._loaded_models['whisper_base'] = self.whisper_model_base

            if model_name == 'whisper_small':
                self.whisper_model_small = ASRInference('Whisper', {'device': self.device, 'model_type': 'small'})
                self._loaded_models['whisper_small'] = self.whisper_model_small

            if model_name == 'whisper_medium':
                self.whisper_model_medium = ASRInference('Whisper', {'device': self.device, 'model_type': 'medium'})
                self._loaded_models['whisper_medium'] = self.whisper_model_medium

    def _clean_data(self, data):
        return re.sub('[^а-яА-Я\s]', '', data).lower().strip()

    def _read_reference(self, num):
        with open(f'{self.path2clips}{num}.txt', 'r') as file:
            reference = file.read().split('|')[1]

        return reference
    
    def _get_transcribe(self, model, filename):
        st = time.time()
        text = model.transcribe(filename)
        et = time.time()

        calc_time = et - st

        result = self._clean_data(' '.join(text))

        return calc_time, result
    
    def _pipeline(self, asr, filename):
        speech = convert_wav(filename)

        # Получение выравнивания с помощью ASR-модели
        alignments = asr.forced_align(speech)
        
        # Фильтрация выравнивания с помощью BERT-модели
        filter_model_config = {'model_path': '/home/ubuntu/ml_rewrite/notebooks/profanity/checkpoint-2500', 
                               'device': 'cuda'}
        
        bert = FilterAbstract(model_name='cointegrated/rubert-tiny2',
                              model_config=filter_model_config)
        
        alignments_filter = bert.filter(alignments)
        
        # Вывод результата фильтрации
        return alignments_filter


    def _get_pipeline_timing(self, model, filename):
        st = time.time()
        align = self._pipeline(model, filename)
        et = time.time()

        calc_time = et - st

        return calc_time, align
    
    
    def create_table(self, table_name):
        datatable = []

        for result in self.test_results:
            table_row = {'Model': result['model'], 'WER': result['WER'], 'RTF': result['RTF'], 'device': self.device}
            datatable.append(table_row)
        
        GigaTable = pd.DataFrame(datatable, columns=['Model', 'WER', 'RTF', 'device'])
        GigaTable.to_csv(table_name, index=False, sep=';')
    
        return datatable

    def test_models(self):
        self.test_results = []

        for model_name, model in self._loaded_models.items():
            rtf_lst = []
            timing_lst = []

            pred_lst = []
            refer_lst = []

            # print(f"[TEST_MODELS]: {model_name}")
            if model:
                # print(f"[TEST_MODELS]: inside, {model_name}")

                for i in tqdm(sample(range(0, self.top-1), self.sample_count)):
                    filename = self.path2clips + f'{i}.wav'
                    
                    if self.pipeline:
                        processingTime, prediction = self._get_pipeline_timing(model, filename)
                    else:
                        processingTime, prediction = self._get_transcribe(model, filename)
                        pred_lst.append(prediction)
                        reference = self._read_reference(i)
                        refer_lst.append(reference)

                    audioLength = get_duration(filename)
                    rtf_lst.append(real_time_factor(processingTime, audioLength))

                    timing_lst.append(processingTime)

                    # print(f'\n{model_name} — Progress: {i}/{self.top}\n')

                # print(pred_lst, '|', refer_lst)

                rtf = round(np.array(rtf_lst).mean(), 3)
                
                if self.pipeline:
                    wer = -1     # Не рассчитывается для pipeline
                else:
                    wer = round(get_wer(pred_lst, refer_lst), 3)
                    
                timing_lst = round(np.array(processingTime).mean(), 3) # В СЕКУНДАХ
            
                self.test_results.append({'model': model_name, 'processing_time': timing_lst, 'RTF': rtf, 'WER': wer})
            else:
                # print(f"[TEST_MODELS]: outside, {model_name}")
                continue # Исключение модели из тестирования
    
        return self.test_results
