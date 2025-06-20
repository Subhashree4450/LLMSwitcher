# core/config.py

BATTERY_THRESHOLD_LOW = 20
BATTERY_THRESHOLD_HIGH = 80
CPU_THRESHOLD_HIGH = 75.0
TEMPERATURE_THRESHOLD = 45.0
TOKEN_THRESHOLD_HIGH = 250
TOKEN_THRESHOLD_LOW = 50

ALPHA = 0.5  # EMA update weight
BOOST_INTERVAL = 5  # How often MLFQ boosts lower-priority models
EPSILON = 0.1
AGING_FACTOR = 3  # or whatever number fits your strategy

# Model identifiers
DISTILBERT = 0
PHI = 1
LLAMA = 2
GEMMA = 3

MODEL_NAMES = {
    DISTILBERT: "DistilBERT",
    PHI: "Phi",
    LLAMA: "LLaMA",
    GEMMA: "Gemma"
}

# Mode flags
class Mode:
    SINGLE = "SINGLE"
    NAIVE = "NAIVE"
    RR = "RR"
    ECOMLS = "ECOMLS"
    MLFQ = "MLFQ"
