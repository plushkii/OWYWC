from asr_inference import ASRInference
from filter_abstract import FilterAbstract
from utils import convert_wav

from huggingface_hub import HfApi, ModelFilter
print(ModelFilter)


SPEECH_FILE = "data_for_tests/test_60.wav"
speech = convert_wav(SPEECH_FILE)

# Получение выравнивания с помощью ASR-модели
model_config = {
            'model_weights': '../ml/ctc_model_weights.ckpt',
            'device': 'cpu', 'batch_size': 1, 'model_type': 'CTC',
            'model_config': './conf/ctc_model_config.yaml'}

asr = ASRInference('GigaAM', model_config)
alignments = asr.forced_align(speech)

# Фильтрация выравнивания с помощью BERT-модели
filter_model_config = {'model_path': '/recognize_words-parasites_model/checkpoint-17303', 
                       'device': 'cuda'}

bert = FilterAbstract(model_name='ai-forever/ruBert-base',
                      model_config=filter_model_config)

alignments_filter = bert.filter(alignments)

# Вывод результата фильтрации
print(alignments)

