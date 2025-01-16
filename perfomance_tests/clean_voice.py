import time
import json
import numpy as np

from asr_inference import ASRInference
from utils import get_wer


PATH = 'datatest/ru/wav_clips/'
DEVICE = 'cuda' # cpu
WER_GATE = 0.8 # Порог очистки, насколько плохие аудиозаписи оставляем. 0.8 - оптимальный параметр
CLEAN_PATH = f'datatest/ru/clean_clips_{WER_GATE}/'


def sec_to_hms(num):
    num = num%(24*60*60) # 86_400 s
    h = num//3600
    temp = num%3600
    m = temp//60
    m = '0'*int(not m//10) + str(m)
    s = temp%60
    s = '0'*int(not s//10) + str(s)
    return f"{h}:{m}:{s}"

def clean_data(data):
    return re.sub('[^а-яА-Я\s]', '', data).lower().strip()

# GigaAM
model_config = {
                'model_config': 'conf/ctc_model_config.yaml',
                'model_weights': '../ml/ctc_model_weights.ckpt',
                'device': DEVICE, 
                'batch_size': 1,
                'model_type': 'CTC'
                }

GigaAM_model = ASRInference('GigaAM', model_config)

def GigaAM_func(model, filename): 
    st = time.time()
    text = model.transcribe(filename)
    et = time.time()

    calc_time = et-st

    result = ' '.join(text)
    
    return calc_time, result

with open(PATH + 'datatest.json', 'r') as file:
    datatest = json.loads(file.read())

num = 0
counter = 0
est_time = []

for filename, refer in datatest.items():
    if num > -1: # Для возможности добавить чекпоинты
        filename = PATH + f'{filename.split(".")[0]}.wav'
        refer = clean_data(refer)
        for _ in range(3):
            calc_time, pred = GigaAM_func(GigaAM_model, filename)
            #print(f'!!!DEBUG_PRED!!!: {type(pred), len(pred), pred, list(pred)}')
            if pred != ' ':
                break
            else:
                continue
    
        est_time.append(int(calc_time*9722))
    
        if pred != ' ':
            wer_score = get_wer([pred], [refer])
        else:
            wer_score = float('inf')
            pred = 'ERROR!'
    
        if wer_score < WER_GATE:
            with open(filename, 'rb') as file:
                music = file.read()
            with open(f'{CLEAN_PATH}{num}.wav', 'wb') as file:
                file.write(music)
            with open(f'{CLEAN_PATH}{num}.txt', 'w') as file:
                file.write(f'{str(num)}|{refer}')
    
            num += 1
        
        counter += 1
        estimated_time = int(np.array(est_time).mean())
        
        print(f'\n\nSpeech_name: {filename}\nProgress: {counter}/9722\nEstimated time: {sec_to_hms(estimated_time)}\nClean: {num}\nWER_score: {round(wer_score, 5)}\n')
        print(f'Prediction: {pred}\nReference: {refer}\n\n')
    else:
        num += 1

print(f'\n\nClean: {num}/9722')
