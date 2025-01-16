import pandas as pd
import json

PATH = "datatest/ru/"
WAV_PATH = "datatest/ru/wav_clips"

invalidated = pd.read_csv('invalidated.tsv', sep='\t')
other = pd.read_csv('other.tsv', sep='\t')
validated = pd.read_csv('validated.tsv', sep='\t')

paths = list(invalidated['path']) + list(other['path']) + list(validated['path'])
sentences = list(invalidated['sentence']) + list(other['sentence']) + list(validated['sentence'])

result = {}
for path, sentence in zip(paths, sentences):
    result[path] = sentence

with open(PATH + 'datatest.json', 'w') as file:
    file.write(json.dumps(result))

with open(PATH + 'datatest.json', 'r') as file:
    datatest = json.loads(file.read())
    
for speech in datatest.keys():
    convert_wav(WAV_PATH + speech)
