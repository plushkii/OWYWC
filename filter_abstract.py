import json
import logging
from transformers import pipeline
from collections import Counter


PUNCTUATION_MARKS = [',', '.', '!', '?', '-', ':', '"', "'"]

TRASH_WORDS = ["вроде", "таким образом", "вообще-то", "то есть", "как бы", 
               "пипец","капец", "офигеть", "нихрена", "нифига себе", "наверное", 
               "ах", "ох", "эх", "ух", "ай", "ой", "ну", "ага", "эй", "фу", "брр"
               "хм", "пф", "тьфу", "ура", "увы", "ишь", "эге", "ба", "нате"]

DEVICE_DEFAULT = 'cpu'


class FilterAbstract:
    def __init__(self, model_config: dict, model_name: str = 'cointegrated/rubert-tiny2'):
        self._model_name = model_name
        self._model_config = model_config

        logging.info('Класс FilterAbstract был запущен')

        self._load_config()
        logging.info('Конфиг загружен и инициализирован')
        
        self._load_model()
        logging.info('Модель загружена')


    def _load_model(self):
        """Метод для подгрузки BERT-модели."""
        if self._model_name == 'cointegrated/rubert-tiny2':
            cuda_device = 0 if self._device == 'cuda' else -1
            self._model_classifier = pipeline('ner', model=self._model_config['model_path'], device=cuda_device)


    def _load_config(self):
        """Метод для загрузки и инициализации конфигурационных
        переменных модели."""        
        
        if 'device' in self._model_config.keys():
            self._device = self._model_config['device']
        else:
            self._device = DEVICE_DEFAULT


    def _profanity_trash_filter(self, alignments: list[list], bad_words: dict) -> list[int]:
        """Метод для эвристической фильтрации по ненормативной
         лексике."""

        marked_positions = []

        for i, token_with_time in enumerate(alignments):
            if i not in marked_positions:
                if token_with_time[0] in bad_words.keys() or token_with_time[0] in TRASH_WORDS:
                    marked_positions.append(i)
                
                elif i + 1 != len(alignments):
                    if str(' '.join([token_with_time[0], alignments[i + 1][0]]).strip()) in TRASH_WORDS:
                        print('marked')
                        marked_positions.extend([i, i + 1])

        return marked_positions


    def _join_tokens(self, alignments: list[list]) -> str:
        """Метод для склейки токенов в один текст."""

        alignments_text = []

        for alignment in alignments:
            alignments_text.append(alignment[0])

        alignments_text = ' '.join(alignments_text)

        return alignments_text


    def _merge_classifier_output(self, classifier_results: list[dict]) -> list[dict]:
        """Метод для склейки выходов BERT, который соединяет разбитые 
        на несколько частей слова, а также убирает знаки
        препинания: [{'word': 'сделан'}, {'word': '##о'}] -> [{'word': 'сделано'}]."""

        classifier_results_merge = [] 
        current_word = ''
        current_entity = []

        for i in range(0, len(classifier_results)):
            if classifier_results[i]['word'] not in PUNCTUATION_MARKS:
                if i + 1 != len(classifier_results):
                    if classifier_results[i + 1]['word'].startswith('#'):
                        current_word += classifier_results[i]['word'].replace('#', '')
                        current_entity.append(classifier_results[i]['entity'])

                    elif current_word != '':
                        current_word += classifier_results[i]['word'].replace('#', '')
                        current_entity.append(classifier_results[i]['entity'])
                        classifier_results_merge.append({'word': current_word, 'entity': Counter(current_entity).most_common(1)[0][0]})

                        current_word = ''
                        current_entity = []

                    else:
                        classifier_results_merge.append({'word': classifier_results[i]['word'],
                                                        'entity': classifier_results[i]['entity']})
                    

                else:
                    classifier_results_merge.append({'word': classifier_results[i]['word'],
                                                        'entity': classifier_results[i]['entity']})

        return classifier_results_merge


    def filter(self, forced_alignments: list[list[str, float, float]]) -> list[str, float, float, bool]:
        """Метод для фильтрации размеченных токенов с временными метками, пока 
        рассматриваются 2 кейса: ненормативная лексика и мусорные слова."""

        logging.info('Начало фильтрации')
        
        with open("data_for_tests/bad_words.json", "r") as file:
            bad_words = file.read()
            bad_words = json.loads(bad_words)

        logging.info('Подгружен список ненормативной лексики')
        
        marked_positions = []
        marked_tokens = []
        
        # Ручная фильтрация по матам и эвристика по мусорным словам
        marked_positions.extend(self._profanity_trash_filter(forced_alignments, bad_words))

        logging.info('Фильтрация по ненормативной лексике завершена')

        # ML фильтрация по мусорным словам
        if self._model_name == 'cointegrated/rubert-tiny2':
            alignments_text = self._join_tokens(forced_alignments)
            classifier_results = self._model_classifier(alignments_text)

            classifier_results_merge = self._merge_classifier_output(classifier_results)

            for i, result in enumerate(classifier_results_merge):
                if result not in marked_positions and result['entity'] != '0':
                    marked_positions.append(i)

            logging.info('Фильтрация по мусорным словам завершена')

        trash_counter = 0

        # Объеденение результатов фильтрации        
        for i, alignment in enumerate(forced_alignments):
            is_trash = True if i in marked_positions else False
            marked_tokens.append([alignment[0], alignment[1], alignment[2], is_trash])

            if is_trash:
                trash_counter += 1

        logging.info(f'Количество маркированных токенов: {trash_counter} из всех {len(classifier_results_merge)} токенов')
        
        logging.info('Фильтрация окончена')

        return marked_tokens