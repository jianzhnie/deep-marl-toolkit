from abc import ABC


class BaseAgent(ABC):

    def reset_agent(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError
