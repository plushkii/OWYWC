from typing import Optional
from model_abstract import ModelAbstract


class ASRInference:
    def __init__(self, model_name: str, model_config: dict):
        self.model = ModelAbstract(model_name, model_config)

    def transcribe(self, speech_filename: str) -> Optional[list[list[str]]]:
        return self.model.transcribe(speech_filename)

    def forced_align(self, speech_filename: str) -> list[list]:
        return self.model.forced_align(speech_filename)