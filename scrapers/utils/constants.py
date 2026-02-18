"""
Shared constants for fault codes, OBD data, and vehicle context.

Used by extractors, validation pipeline, and process_scraped_data.py.
"""
import re

# Fault code patterns - extraction (find codes in text)
FAULT_CODE_EXTRACT_PATTERNS = [
    re.compile(r"\b(P[0-9][0-9][0-9][0-9])\b", re.IGNORECASE),
    re.compile(r"\b([A-Z][0-9][A-Z][0-9][0-9])\b"),
    re.compile(r"\b([A-Z][0-9][0-9][0-9][0-9])\b"),
]

# Fault code patterns - validation (match full string)
FAULT_CODE_VALIDATE_PATTERNS = [
    re.compile(r"^P[0-9][0-9][0-9][0-9]$"),
    re.compile(r"^[A-Z][0-9][A-Z][0-9][0-9]$"),
    re.compile(r"^[A-Z][0-9][0-9][0-9][0-9]$"),
]

# OBD-II parameter ranges (for validation)
OBD_RANGES = {
    "engine_rpm": (0, 8000),
    "coolant_temp": (-40, 150),
    "throttle_position": (0, 100),
    "maf_sensor": (0, 1000),
    "fuel_trim_bank1": (-100, 100),
    "fuel_trim_bank2": (-100, 100),
    "o2_sensor_bank1": (0, 1.5),
    "o2_sensor_bank2": (0, 1.5),
    "intake_air_temp": (-40, 150),
    "barometric_pressure": (0, 200),
    "timing_advance": (-50, 50),
}

# OBD parameter extraction patterns
OBD_PARAM_PATTERNS = {
    "engine_rpm": re.compile(
        r"(?:engine\s*rpm|rpm)\s*[=:]\s*(\d+(?:\.\d+)?)|(\d+)\s*rpm",
        re.IGNORECASE,
    ),
    "coolant_temp": re.compile(
        r"(?:coolant\s*temp(?:erature)?|coolant)\s*[=:]\s*(\d+(?:\.\d+)?)|(\d+)\s*[°cC]",
        re.IGNORECASE,
    ),
    "throttle_position": re.compile(
        r"(?:throttle\s*position|tps)\s*[=:]\s*(\d+(?:\.\d+)?)|(\d+)\s*%",
        re.IGNORECASE,
    ),
    "maf_sensor": re.compile(
        r"(?:maf|mass\s*air\s*flow)\s*[=:]\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*g/s",
        re.IGNORECASE,
    ),
    "fuel_trim_bank1": re.compile(
        r"(?:fuel\s*trim\s*bank\s*1|stft\s*b1|ltft\s*b1)\s*[=:]\s*([+-]?\d+(?:\.\d+)?)|([+-]?\d+(?:\.\d+)?)\s*%",
        re.IGNORECASE,
    ),
    "fuel_trim_bank2": re.compile(
        r"(?:fuel\s*trim\s*bank\s*2|stft\s*b2|ltft\s*b2)\s*[=:]\s*([+-]?\d+(?:\.\d+)?)|([+-]?\d+(?:\.\d+)?)\s*%",
        re.IGNORECASE,
    ),
    "o2_sensor_bank1": re.compile(
        r"(?:o2\s*bank\s*1|oxygen\s*sensor\s*b1)\s*[=:]\s*(\d+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
    "o2_sensor_bank2": re.compile(
        r"(?:o2\s*bank\s*2|oxygen\s*sensor\s*b2)\s*[=:]\s*(\d+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
    "intake_air_temp": re.compile(
        r"(?:intake\s*air\s*temp|iat)\s*[=:]\s*(\d+(?:\.\d+)?)|(\d+)\s*[°cC]",
        re.IGNORECASE,
    ),
}

# Vehicle context
VEHICLE_MAKES = {
    "bmw", "mercedes", "audi", "volkswagen", "vw", "porsche", "mini",
    "ford", "chevrolet", "chevy", "dodge", "jeep", "toyota", "honda",
    "nissan", "mazda", "subaru", "hyundai", "kia", "volvo", "lexus",
    "acura", "infiniti", "cadillac", "buick", "gmc", "ram",
}
VEHICLE_MAKE_PATTERN = re.compile(
    r"\b(" + "|".join(VEHICLE_MAKES) + r")\b",
    re.IGNORECASE,
)
YEAR_PATTERN = re.compile(r"\b(19[89]\d|20[0-2]\d)\b")
ENGINE_PATTERN = re.compile(
    r"\b(N54|N55|B58|B48|S55|S58|M54|M52|2JZ|1JZ|LS[0-9]|VQ[0-9]{2})\b",
    re.IGNORECASE,
)
MILEAGE_PATTERN = re.compile(
    r"(?:mileage|miles|km|odometer)\s*[=:]\s*([\d,]+)|([\d,]+)\s*(?:k\s*miles?|miles?|km)",
    re.IGNORECASE,
)
