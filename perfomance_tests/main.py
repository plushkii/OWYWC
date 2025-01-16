from models_perfomance_test import ModelsPerfomanceTest


DEVICE = 'cpu'      # cuda
TOP = 500           # Сколько у вас данных, или первые значения
SAMPLE_COUNT = 50   # Кол-во случайных примеров из топа
TABLE_NAME = 'ResultTable.csv'


# Список всех доступных моделей
models_lst = ['vosk-model-small-ru-0.22', 'GigaAM_CTC', 'whisper_tiny',
              'whisper_base', 'whisper_small', 'whisper_medium']

if __name__ == "__main__":
    test = ModelsPerfomanceTest(models_lst, DEVICE,
                                TOP, SAMPLE_COUNT, pipeline=False)
    print(test.test_models())
    test.create_table(TABLE_NAME)
