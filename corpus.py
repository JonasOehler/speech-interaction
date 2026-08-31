from abc import ABC, abstractmethod
from datasets import load_dataset_builder, load_dataset

class AbstractCorpus(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_speech_corpus_details(self) -> str:
        pass

    def load_speech_corpus(self):
        pass

class Librispeech(AbstractCorpus):
    def __init__(self):
        super().__init__()

    def get_speech_corpus_details(self) -> str:
        builder = load_dataset_builder("openslr/librispeech_asr", "clean")
        return builder.info

    def load_speech_corpus(self):
        return load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True)


class Fleurs(AbstractCorpus):
    def __init__(self):
        super().__init__()

    def get_speech_corpus_details(self) -> str:
        builder = load_dataset_builder("google/fleurs", "en_us")
        return builder.info

    def load_speech_corpus(self):
        return load_dataset("google/fleurs", "en_us", split="test", streaming=True)


class Peoples_Speech(AbstractCorpus):
    def __init__(self):
        super().__init__()

    def get_speech_corpus_details(self) -> str:
        builder = load_dataset_builder("MLCommons/peoples_speech", "clean")
        return builder.info

    def load_speech_corpus(self):
        return load_dataset("MLCommons/peoples_speech", "clean", split="test", streaming=True)
    
class GigaSpeech(AbstractCorpus): #This is a gated dataset
    def __init__(self):
        super().__init__()

    def get_speech_corpus_details(self) -> str:
        builder = load_dataset_builder("speechcolab/gigaspeech", "test")
        return builder.info

    def load_speech_corpus(self):
        return load_dataset("speechcolab/gigaspeech", "test", split="test", streaming=True)

def create_corpus(args) -> AbstractCorpus:
    if args.speech_corpus == 'librispeech':
        return Librispeech()
    if args.speech_corpus == 'fleurs':
        return Fleurs()
    if args.speech_corpus == 'peoples_speech':
        return Peoples_Speech()
    if args.speech_corpus == 'gigaspeech':
        return GigaSpeech()