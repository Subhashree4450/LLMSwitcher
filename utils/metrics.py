# utils/metrics.py

import random

def get_battery_level():
    # Placeholder: on Pi or laptop you can link this with psutil
    return random.randint(20, 90)

def get_cpu_usage():
    return round(random.uniform(10.0, 90.0), 2)

def get_temperature():
    return round(random.uniform(30.0, 60.0), 2)
