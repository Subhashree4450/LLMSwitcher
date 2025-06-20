import random
from core.config import *
from utils.metrics import get_battery_level, get_cpu_usage, get_temperature

class EcoModel:
    def __init__(self, model_id, name, energy=1.5, confidence=0.8):
        self.model_id = model_id
        self.name = name
        self.energy_consumption = energy
        self.confidence_score = confidence
        self.execution_time = 0.0
        self.ema_score = 0.5
        self.execution_count = 0

    def update_performance(self, new_energy, new_confidence, new_time):
        c = self.execution_count
        self.energy_consumption = (self.energy_consumption * c + new_energy) / (c + 1)
        self.confidence_score = (self.confidence_score * c + new_confidence) / (c + 1)
        self.execution_time = (self.execution_time * c + new_time) / (c + 1)
        self.execution_count += 1

class EpsilonGreedySelector:
    def __init__(self):
        self.epsilon = EPSILON
        self.models = [
            EcoModel(DISTILBERT, "DistilBERT", 1.5, 0.8),
            EcoModel(PHI, "Phi", 1.8, 0.85),
            EcoModel(LLAMA, "LLaMA", 2.0, 0.9),
            EcoModel(GEMMA, "Gemma", 2.2, 0.95),
        ]

    def calculate_score(self, battery, energy, temp, cpu, token_factor, confidence, e_factor):
        weight_battery = 0.4 if battery < BATTERY_THRESHOLD_LOW else 0.2
        weight_cpu = 0.4 if temp > TEMPERATURE_THRESHOLD else 0.2
        weight_conf = 1.0 - weight_battery - weight_cpu

        battery_score = 1.0 - (energy / 5.0)
        cpu_score = 1.0 - (cpu / 100.0)

        score = (battery_score * weight_battery + cpu_score * weight_cpu + confidence * weight_conf)
        return e_factor * token_factor * score

    def select_model(self):
        battery = get_battery_level()
        cpu = get_cpu_usage()
        temp = get_temperature()

        if random.random() < self.epsilon:
            return random.choice(self.models).model_id

        best_model = None
        best_score = float("inf")

        for model in self.models:
            score = self.calculate_score(
                battery, model.energy_consumption, temp, cpu,
                1.0, model.confidence_score, 1.0
            )
            if score < best_score:
                best_score = score
                best_model = model

        return best_model.model_id