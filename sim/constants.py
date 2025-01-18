from enum import Enum

class DiseaseState(Enum):
    LUNG_CA = 0
    COPD = 1
    PNEUMONIA = 2
    BRONCHITIS = 3
    BRONCHIECTASIS = 4
    OTHER = 5
    
class LungCancerStage(Enum):
    NONE = 0
    CIS = 1 
    STAGE_I = 2
    STAGE_II = 3
    STAGE_III = 4
    STAGE_IV = 5
    REMISSION = 6


MIMIC_AGE_COUNTS = [
    244.0,
    384.0,
    363.0,
    536.0,
    613.0,
    516.0,
    875.0,
    1252.0,
    1255.0,
    2221.0,
    2551.0,
    2258.0,
    3292.0,
    3392.0,
    2456.0,
    3111.0,
    2918.0,
    1981.0,
    2174.0,
    2404.0,
]

MIMIC_AGE_BINS = [
    18.0,
    21.65,
    25.3,
    28.95,
    32.6,
    36.25,
    39.9,
    43.55,
    47.2,
    50.85,
    54.5,
    58.15,
    61.8,
    65.45,
    69.1,
    72.75,
    76.4,
    80.05,
    83.7,
    87.35,
    91.0,
]

LUNG_CANCER_TRANSITIONS = {
    0: {
        0: 0.5,
        1: 0.5,
        2: 0.5,
    }
    
}