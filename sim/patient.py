from dataclasses import dataclass
import numpy as np
from typing import List, Tuple
from enum import Enum

from .constants import MIMIC_AGE_COUNTS, MIMIC_AGE_BINS, LUNG_CANCER_TRANSITIONS, LungCancerStage
from .utils import sample_from_distribution

@dataclass
class RiskFactors:

    smoker: bool = False
    work_exposure: bool = False
    low_ses: bool = False
    pmh: bool = False
    fh: bool = False

def sample_age_from_mimic():
    """Generate age based on the MIMIC population data"""
    
    # Normalize counts to probabilities
    probs = MIMIC_AGE_COUNTS / np.sum(MIMIC_AGE_COUNTS)

    # Select bin
    bin_idx = np.random.choice(len(MIMIC_AGE_COUNTS), p=probs)

    # Generate uniform age within selected bin
    min_age = MIMIC_AGE_BINS[bin_idx]
    max_age = MIMIC_AGE_BINS[bin_idx + 1]

    return np.random.uniform(min_age, max_age)

def init_risk_population(
    n_patients: int,
    high_risk_proportion: float = 0.2,
    high_risk_cancer_rate: float = 0.5,
    base_cancer_rate: float = 0.05,
) -> List[Tuple[int, bool, bool]]:
    """
    Initialize a risk stratified patient population positive / negative for cancer
    using real age distribution data

    Args:
        n_patients (int): Number of patients to generate
        high_risk_proportion (float, optional): Proportion of population that is high risk
        for lung cancer. Defaults to 0.2.
        high_risk_cancer_rate (float, optional): Rate of lung cancer in high risk population. Defaults to 0.5.
        base_cancer_rate (float, optional): Rate of lung cancer in base population. Defaults to 0.05.

    Returns:
        List[Tuple[int, bool, bool]]: List of (index age, has cancer, high risk)
    """
    # Calculate numbers
    n_high_risk = int(n_patients * high_risk_proportion)
    n_normal_risk = n_patients - n_high_risk

    n_cancer_high_risk = int(n_high_risk * high_risk_cancer_rate)
    n_cancer_normal_risk = int(n_normal_risk * base_cancer_rate)

    population = []

    # Generate high-risk group
    high_risk_cancer_ages = []
    for _ in range(n_cancer_high_risk):
        age = sample_age_from_mimic()
        # Bias towards older age for cancer cases
        while age < 45 or np.random.random() < 0.7:
            age = sample_age_from_mimic()
        high_risk_cancer_ages.append(age)

    high_risk_normal_ages = [
        sample_age_from_mimic()
        for _ in range(n_high_risk - n_cancer_high_risk)
    ]

    # Add high-risk patients
    for age in high_risk_cancer_ages:
        population.append((int(age), True, True))
    for age in high_risk_normal_ages:
        population.append((int(age), False, True))

    # Generate normal-risk group
    normal_risk_cancer_ages = []
    for _ in range(n_cancer_normal_risk):
        age = sample_age_from_mimic()
        # Bias towards older age for cancer cases
        while age < 50 or np.random.random() < 0.7:
            age = sample_age_from_mimic()
        normal_risk_cancer_ages.append(age)

    normal_risk_normal_ages = [
        sample_age_from_mimic()
        for _ in range(n_normal_risk - n_cancer_normal_risk)
    ]

    # Add normal-risk patients
    for age in normal_risk_cancer_ages:
        population.append((int(age), True, False))
    for age in normal_risk_normal_ages:
        population.append((int(age), False, False))

    # Shuffle the population
    np.random.shuffle(population)

    return population


class Symptom:

    def __init__(self, name, emission):
        pass

    def sample(self):
        print('hello')

class Patient:

    cancer: bool
    timestep: int
    age: int
    gender: int
    condition: int
    cancer_stage: LungCancerStage

    def __init__(self, index_age, has_cancer, high_risk):
        
        # Assign gender - high risk has 1.1x male, low risk with cancer 0.9x male, otherwise 5050
        
        if high_risk:
            self.gender = np.random.choice([0, 1], p=[0.45, 0.55])
        else:
            self.gender = np.random.randint(2)
        
        self.age = index_age
        
        if has_cancer:
            self.cancer_stage = LungCancerStage.CIS
        
        pass

    def next_step(self):

        if self.cancer:
            pass

        else:
            pass


    def progress_cancer_stage(self):
        if not self.cancer: pass
        
        self.cancer_stage = sample_from_distribution(LUNG_CANCER_TRANSITIONS[self.cancer_stage])

    