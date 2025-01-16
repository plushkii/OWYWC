from asr_inference import ASRInference
from utils import convert_wav, filter
from main import model_config


def test_transcribe_normal():
    
    """кейс с саморучно записанным нормальным аудио, ETALON подобран саморучно"""
    
    ETALON_OUTPUT = ["здравствуйте", "мы", "начинаем", "наш", "подкаст", "по", "обучению", "модели", "йоло", "восемь"]

    SPEECH_FILE = "data_for_tests/test_transcribition_normal.wav"
    speech = convert_wav(SPEECH_FILE)
    
    model = ASRInference('GigaAM', model_config)
    output = model.transcribe(speech)
    assert output == ETALON_OUTPUT


def test_transcribe_with_waste_words():
    
    """кейс с саморучно записанным аудио с мусорными словами, ETALON подобран саморучно"""
    
    ETALON_OUTPUT = ["ну", "типа", "что", "то", "типа", "хамстер", "бизнес"]

    SPEECH_FILE = "data_for_tests/test_transcribition_with_waste_words.wav"
    speech = convert_wav(SPEECH_FILE)

    model = ASRInference('GigaAM', model_config)
    output = model.transcribe(speech)

    assert output == ETALON_OUTPUT
    

def test_transcribe_normal_10sec():
    
    """кейс со скачанным 10 секундным аудио, в качестве ETALON'а выход модели GigaAM"""
    
    ETALON_OUTPUT = ['то', 'как', 'мы', 'его', 'воспринимаем', 'нравится', 'он', 'нам', 'или', 'нет', 'кажется', 'он', 'песклявым', 'или', 'наоборот', 'бархатным',
                     'глубоким', 'и', 'красивым', 'это', 'наверное', 'то', 'что', 'больше', 'всего', 'влияет', 'на', 'нашего']


    SPEECH_FILE = "data_for_tests/test_10.wav"
    speech = convert_wav(SPEECH_FILE)

    model = ASRInference('GigaAM', model_config)
    output = model.transcribe(speech)

    assert output == ETALON_OUTPUT


def test_transcribe_normal_60sec():
    
    """кейс со скачанным 60 секундным аудио, в качестве ETALON'а выход модели GigaAM"""
    
    ETALON_OUTPUT = ['то', 'как', 'мы', 'его', 'воспринимаем', 'нравится', 'он', 'нам', 'или', 'нет', 'кажется', 'он', 'песклявым', 'или', 'наоборот', 'бархатным',
                     'глубоким', 'и', 'красивым', 'это', 'наверное', 'то', 'что', 'больше', 'всего', 'влияет', 'на', 'наше', 'восприятие', 'голоса', 'того', 'или',
                     'иного', 'человека', 'именно', 'поэтому', 'в', 'сегодняшнем', 'выпуске', 'мы', 'вместе', 'с', 'вами', 'постараемся', 'ответить', 'на', 'вопросы',
                     'можно', 'ли', 'изменить', 'тот', 'тембр', 'голоса', 'который', 'дан', 'нам', 'при', 'рождении', 'и', 'если', 'можно', 'то', 'как', 'его', 'улучшить',
                     'как', 'сделать', 'его', 'более', 'красивым', 'и', 'приятным', 'клубничкой', 'начнем', 'с', 'ответа', 'на', 'первый', 'вопрос', 'можем', 'ли', 'мы',
                     'изменить', 'тот', 'тембр', 'голоса', 'который', 'дан', 'нам', 'при', 'рождении', 'для', 'этого', 'нам', 'естественно', 'необходимо', 'разобраться',
                     'в', 'процессе', 'звукоизлечения', 'то', 'есть', 'как', 'вообще', 'формируется', 'наш', 'тембр', 'голоса', 'на', 'картинке', 'вы', 'видите', 'важнейшую',
                     'часть', 'нашего', 'голосового', 'аппарата', 'который', 'называется', 'голосовые', 'складки', 'именно', 'благодаря', 'колебанию', 'этих', 'складок', 'и',
                     'рождается', 'звуковая', 'волна', 'которая']

    SPEECH_FILE = "data_for_tests/test_60.wav"
    speech = convert_wav(SPEECH_FILE)

    model = ASRInference('GigaAM', model_config)
    output = model.transcribe(speech)

    assert output == ETALON_OUTPUT