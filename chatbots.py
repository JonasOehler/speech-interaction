from abc import ABC, abstractmethod
import sys

class AbstractChatbot(ABC):

    def __init__(self, skills={}) -> None:
        self.init_speech = None
        self.skills = skills

    @abstractmethod
    def respond(self, user_input: str) -> str:
        '''Generate an answer with given user input.'''
        pass

class ElizaChatbot(AbstractChatbot):
    
    def __init__(self, skills: dict, path):
        super().__init__(skills)
        
        sys.path.append(path)
        import eliza

        self.eliza = eliza.Eliza()
        self.eliza.load(f'{path}/doctor.txt')
        self.init_speech = self.eliza.initial()
    
    def respond(self, user_input: str) -> str:
        response = self.eliza.respond(user_input)
        if response.startswith('function='):
            fn = self.skills.get(response[len('function='):], None)
            response = 'I cannot do that.'
            if fn is not None:
                response = fn()
        if response is None:
            response = self.eliza.final()
        return response

def create_chatbot(skills, args) -> AbstractChatbot:
    if args.chatbot_type == 'eliza':
        return ElizaChatbot(skills, args.eliza_path)