from abc import ABC, abstractmethod


class BaseAgent(ABC):

    @abstractmethod
    def reset_agent(self):
        raise NotImplementedError

    @abstractmethod
    def sample(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError

    @abstractmethod
    def learn(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def restore(self):
        raise NotImplementedError
