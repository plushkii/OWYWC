from asr_inference import ASRInference
from utils import convert_wav, filter
from main import model_config


def test_filter_with_one_arg():
    
    """кейс с одним переданным аргументом, ETALON выдуман"""
    
    ETALON_OUTPUT = [['Мама', 0.000, 0.680, False]]

    # model = ASRInference('GigaAM', model_config)
    
    tokens_timing = [['Мама', 0.000, 0.680]]
    output = filter(tokens_timing)
    assert output == ETALON_OUTPUT


def test_filter_with_repeat_letter():
    
    """таких кейсов не обнаружено, ETALON выдуман"""
    
    ETALON_OUTPUT = [['ну', 0.000, 0.680, True], ['э', 0.823, 1.234, True], ['a', 1.342, 1.700, True],
                     ['наверное', 1.823, 2.699, False], ['да', 2.833, 3.000, False]]

    # model = ASRInference('GigaAM', model_config)

    tokens_timing = [['ну', 0.000, 0.680], ['э', 0.823, 1.234], ['a', 1.342, 1.700],
                     ['наверное', 1.823, 2.699], ['да', 2.833, 3.000]]
    output = filter(tokens_timing)
    assert output == ETALON_OUTPUT
    
    
def test_filter_smart():
    
    """кейс со словом, которое употребляется в контексте отличном от мусорного, ETALON выдуман"""
    
    ETALON_OUTPUT = [['один', 0.000, 0.680, False], ['провод', 0.823, 1.234, False], ['короче', 1.342, 2.110, False],
                     ['другого', 2.199, 2.999, False]]

    # model = ASRInference('GigaAM', model_config)

    tokens_timing = [['один', 0.000, 0.680], ['провод', 0.823, 1.234], ['короче', 1.342, 2.110],
                     ['другого', 2.199, 2.999]]
    output = filter(tokens_timing)
    assert output == ETALON_OUTPUT


def test_filter_with_waste_words():
    
    """кейс с мусорными словами, ETALON выдуман"""
    
    ETALON_OUTPUT = [['ну', 0.000, 0.680, True], ['типа', 0.823, 1.234, True], ['как-то', 1.342, 1.700, False],
                     ['так', 1.823, 2.699, False], ['наверное', 2.833, 3.000, False]]

    # model = ASRInference('GigaAM', model_config)

    tokens_timing = [['ну', 0.000, 0.680], ['типа', 0.823, 1.234], ['как-то', 1.342, 1.700],
                     ['так', 1.823, 2.699], ['наверное', 2.833, 3.000]]
    output = filter(tokens_timing)
    assert output == ETALON_OUTPUT


# def test_filter_with_repeat_words():
    
#     """"кейс с повторяющимися словами (в данный момент не рассматриваем), ETALON выдуман"""
    
#     ETALON_OUTPUT = [['знаешь', 0.000, 0.680, False], ['знаешь', 0.823, 1.234, True],
#                      ['крутая', 1.342, 1.700, False],
#                      ['такая', 1.823, 2.699, False], ['есть', 2.833, 3.000, False]]

    # model = ASRInference('GigaAM', model_config)

    tokens_timing = [['знаешь', 0.000, 0.680], ['знаешь', 0.823, 1234], ['крутая', 1.342, 1.700],
                     ['такая', 1.823, 2.699], ['есть', 2.833, 3.000]]
    output = filter(tokens_timing)
    assert output == ETALON_OUTPUT


def test_filter_normal():
    
    """"кейс с нормальными входными, ETALON выдуман"""
    
    ETALON_OUTPUT = [['да', 0.000, 0.680, False], ['мыши', 0.823, 1.234, False], ['перегрызли', 1.342, 1.700, False],
                     ['провода', 1.823, 2.699, False], ['то', 2.833, 3.000, False], ['есть', 3.111, 4.057, False],  ['ребра', 4.150, 5.101, False]]

    # model = ASRInference('GigaAM', model_config)
    
    tokens_timing = [['да', 0.000, 0.680], ['мыши', 0.823, 1.234], ['перегрызли', 1.342, 1.700],
                     ['провода', 1.823, 2.699], ['то', 2.833, 3.000], ['есть', 3.111, 4.057],  ['ребра', 4.150, 5.101]]
    output = filter(tokens_timing)
    assert output == ETALON_OUTPUT
    

# def test_filter_pause():
    
#     """"кейс для проверки длинных пауз, ETALON саморучный
#         так, паузы нужно переделать в фильтре у них механика немного другая
#         (пауза входит в тайм спенд токена)"""
    
#     ETALON_OUTPUT = [['ну', 0.0, 5.307375, True], ['типа', 5.5069375, 7.1430625, True], ['типа', 7.3425625, 7.7815, True], ['ну', 8.8589375, 10.3354375, True], ['блин', 10.4950625, 12.490375, True]]

#     # model = ASRInference('GigaAM', model_config)
    
#     tokens_timing = [['ну', 0.0, 5.307375, True], ['типа', 5.5069375, 7.1430625, True], ['типа', 7.3425625, 7.7815, True], ['ну', 8.8589375, 10.3354375, True], ['блин', 10.4950625, 12.490375, True]]

#     output = filter(tokens_timing)
#     assert output == ETALON_OUTPUT