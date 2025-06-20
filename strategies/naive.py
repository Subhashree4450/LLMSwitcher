from core.config import *
from utils.metrics import get_battery_level, get_cpu_usage, get_temperature
from utils.tokenizer_utils import get_token_count

class NaiveSelector:
    def select_model(self, content):
        battery = get_battery_level()
        cpu = get_cpu_usage()
        temp = get_temperature()
        tokens = get_token_count(content)

        if temp > TEMPERATURE_THRESHOLD or (battery <= BATTERY_THRESHOLD_LOW and cpu > CPU_THRESHOLD_HIGH):
            return DISTILBERT

        elif tokens >= TOKEN_THRESHOLD_HIGH:
            if battery >= BATTERY_THRESHOLD_HIGH and cpu <= CPU_THRESHOLD_HIGH:
                return GEMMA
            else:
                return PHI

        elif battery >= BATTERY_THRESHOLD_HIGH:
            return PHI if cpu > CPU_THRESHOLD_HIGH else LLAMA

        elif battery > BATTERY_THRESHOLD_LOW:
            return PHI

        else:
            return DISTILBERT
