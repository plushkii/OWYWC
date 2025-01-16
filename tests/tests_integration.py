from asr_inference import ASRInference
from utils import convert_wav, filter
from main import model_config


def pipe_wav(speech_file) -> list[list]:
    
    """полный прогон файла"""

    model = ASRInference('GigaAM', model_config)
    speech = convert_wav(speech_file)
    alignments = model.forced_align(speech)
    alignments_filter = filter(alignments)
    return alignments_filter
    
    

def test_integration():
    
    """тест всего функционала"""

    # подаеи wav в pipe_wav
    # сравниваем с эталоном
    ETALON_OUTPUT = [['ну', 0.0, 5.307375, True], ['типа', 5.5069375, 7.1430625, True], ['типа', 7.3425625, 7.7815, True], ['ну', 8.8589375, 10.3354375, True], ['блин', 10.4950625, 12.490375, True]]

    output = pipe_wav("test_with_parasites.wav")
    print(output)
    assert output == ETALON_OUTPUT
