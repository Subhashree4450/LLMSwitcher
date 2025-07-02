from collections import deque
from core.config import *
from utils.metrics import get_battery_level, get_cpu_usage, get_temperature
from utils.tokenizer_utils import get_token_count

# Model priorities
PRIORITY_HIGH = 0
PRIORITY_MID = 1
PRIORITY_LOW = 2

class MLFQModel:
    def __init__(self, model_id, name, type="Light"):
        self.model_id = model_id
        self.name = name
        self.type = type  # "Light" or "Heavy"
        self.priority = PRIORITY_MID
        self.token_factor = 1.0
        self.e_factor = 1.0
        self.execution_count = 0
        self.times_rejected = 0

class MLFQSelector:
    def __init__(self):
        self.high_queue = deque()
        self.mid_queue = deque()
        self.low_queue = deque()
        self.models = []

        self.ema_score = 0.5
        self.boost_interval = BOOST_INTERVAL
        self.runs_since_boost = 0
        self.prev_model = None

        self._initialize_models()

    def _initialize_models(self):
        # "Light" = efficient, low-resource model
        # "Heavy" = powerful, high-resource model
        self.models = [
            MLFQModel(PHI, "Phi", "Light"),
            MLFQModel(LLAMA, "LLaMA", "Heavy"),
            MLFQModel(GEMMA, "Gemma", "Heavy"),
        ]
        for model in self.models:
            self.mid_queue.append(model)

    def select_model(self, content, user_feedback=1.0, first_question=False):
        model = self._get_from_queues()
        if not model:
            raise ValueError("No models available in queues.")

        token_count = get_token_count(content)
        if model.type == "Heavy" and token_count > TOKEN_THRESHOLD_HIGH:
            model.token_factor = 2.0
        else:
            model.token_factor = 1.0

        model.execution_count += 1
        self.runs_since_boost += 1

        if not first_question and self.prev_model:
            self._adjust_model_priority(user_feedback)

        self.prev_model = model

        if self.runs_since_boost > self.boost_interval:
            self._boost_all()
            self.runs_since_boost = 0

        return model

    def _get_from_queues(self):
        if self.high_queue:
            return self.high_queue[0]
        elif self.mid_queue:
            return self.mid_queue[0]
        elif self.low_queue:
            return self.low_queue[0]
        return None

    def _adjust_model_priority(self, feedback):
        battery = get_battery_level()
        cpu = get_cpu_usage()
        temp = get_temperature()

        score = self._calculate_score(
            battery, cpu, temp,
            self.prev_model.token_factor,
            feedback, self.prev_model.e_factor
        )

        if feedback == 0.0:
            self.prev_model.times_rejected += 1
            if self.prev_model.times_rejected >= 3:
                self._demote(self.prev_model)
                self.prev_model.execution_count = 0
                self.prev_model.times_rejected = 0
        elif score > self.ema_score + 0.1:
            self._promote(self.prev_model)
        elif score < self.ema_score - 0.1:
            self._demote(self.prev_model)

        if self.prev_model.execution_count > (3 - self.prev_model.priority) * AGING_FACTOR:
            self._move_to_queue(self.prev_model)

        self.ema_score = ALPHA * score + (1 - ALPHA) * self.ema_score

    def _calculate_score(self, battery, cpu, temp, token_factor, feedback, e_factor):
        weight_battery = 0.4 if battery < BATTERY_THRESHOLD_LOW else 0.2
        weight_cpu = 0.4 if temp > TEMPERATURE_THRESHOLD else 0.2
        weight_feedback = round(1.0 - weight_battery - weight_cpu, 1)

        battery_score = 1.0 - (battery / 100.0)
        cpu_score = 1.0 - (cpu / 100.0)

        total_score = (
            battery_score * weight_battery +
            cpu_score * weight_cpu +
            feedback * weight_feedback
        )
        return e_factor * token_factor * total_score

    def _promote(self, model):
        if model.priority > PRIORITY_HIGH:
            model.priority -= 1
            self._move_to_queue(model)

    def _demote(self, model):
        if model.priority < PRIORITY_LOW:
            model.priority += 1
            self._move_to_queue(model)

    def _move_to_queue(self, model):
        self.high_queue = deque([m for m in self.high_queue if m != model])
        self.mid_queue = deque([m for m in self.mid_queue if m != model])
        self.low_queue = deque([m for m in self.low_queue if m != model])

        if model.priority == PRIORITY_HIGH:
            self.high_queue.append(model)
        elif model.priority == PRIORITY_MID:
            self.mid_queue.append(model)
        else:
            self.low_queue.append(model)

    def _boost_all(self):
        while self.mid_queue:
            model = self.mid_queue.popleft()
            model.priority = PRIORITY_HIGH
            model.execution_count = 0
            model.times_rejected = 0
            self.high_queue.append(model)

        while self.low_queue:
            model = self.low_queue.popleft()
            model.priority = PRIORITY_HIGH
            model.execution_count = 0
            model.times_rejected = 0
            self.high_queue.append(model)
