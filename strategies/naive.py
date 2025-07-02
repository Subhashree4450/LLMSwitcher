from core.config import *
from utils.metrics import get_battery_level, get_cpu_usage, get_temperature
from utils.tokenizer_utils import get_token_count

class NaiveSelector:
    def select_model(self, content):
        battery = get_battery_level()
        cpu = get_cpu_usage()
        temp = get_temperature()
        tokens = get_token_count(content)

        # Case 1: High temperature or low battery + high CPU
        if temp > TEMPERATURE_THRESHOLD or (battery <= BATTERY_THRESHOLD_LOW and cpu > CPU_THRESHOLD_HIGH):
            return PHI  # lightweight and lower resource use

        # Case 2: High token count (needs strong model)
        elif tokens >= TOKEN_THRESHOLD_HIGH:
            if battery >= BATTERY_THRESHOLD_HIGH and cpu <= CPU_THRESHOLD_HIGH:
                return GEMMA  # powerful model for long input
            else:
                return PHI  # fallback to lighter model

        # Case 3: Battery is high
        elif battery >= BATTERY_THRESHOLD_HIGH:
            return PHI if cpu > CPU_THRESHOLD_HIGH else LLAMA

        # Case 4: Battery is medium
        elif battery > BATTERY_THRESHOLD_LOW:
            return PHI

        # Case 5: Battery is critically low (fallback)
        else:
            return PHI
