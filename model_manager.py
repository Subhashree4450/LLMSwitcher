# core/model_manager.py

from config import *
from strategies.naive import NaiveSelector
from strategies.rr import RoundRobinSelector
from strategies.ecomls import EpsilonGreedySelector
from strategies.mlfq import MLFQSelector

class ModelManager:
    def __init__(self):
        self.mode = Mode.MLFQ
        self.adapt = True
        self.model_to_use = DISTILBERT

        self.naive_selector = NaiveSelector()
        self.rr_selector = RoundRobinSelector()
        self.eco_selector = EpsilonGreedySelector()
        self.mlfq_selector = MLFQSelector()

        self.prev_mode = None
        self.prev_model = None
        self.runs_since_boost = 0

    def select_model(self, content, user_feedback=1.0, first_question=False):
        if not self.adapt and self.mode == Mode.SINGLE:
            return self.model_to_use

        if self.mode == Mode.MLFQ:
            model = self.mlfq_selector.select_model(content, user_feedback, first_question)
            self.prev_mode = Mode.MLFQ
            return model

        elif self.mode == Mode.NAIVE:
            model = self.naive_selector.select_model(content)
            self.prev_mode = Mode.NAIVE
            return model

        elif self.mode == Mode.RR:
            model = self.rr_selector.get_next_model()
            self.prev_mode = Mode.RR
            return model

        elif self.mode == Mode.ECOMLS:
            model = self.eco_selector.select_model()
            self.prev_mode = Mode.ECOMLS
            return model

        return self.model_to_use
