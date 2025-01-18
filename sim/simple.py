import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import itertools


@dataclass
class PatientVisit:
    patient_id: int
    visit_id: int
    clinical_state: np.ndarray
    interventions: np.ndarray
    diagnoses: np.ndarray
    note: str
    time_to_next: int  # months until next visit (0 = death, 13 = no visit in 12 months)

    def to_dict(self):
        clinical_dict = dict(
            patient_id=self.patient_id,
            visit_id=self.visit_id,
        )

        clinical_dict.update({f"so_{i}": x for i, x in enumerate(self.clinical_state)})
        clinical_dict.update({f"ix_{i}": x for i, x in enumerate(self.interventions)})
        clinical_dict.update({f"dx_{i}": x for i, x in enumerate(self.diagnoses)})

        clinical_dict.update(
            dict(
                note=self.note,
                time_to_next=self.time_to_next,
            )
        )

        return clinical_dict


@dataclass
class PatientTrajectory:
    patient_id: int
    visits: List[PatientVisit]

    def __len__(self):
        return len(self.visits)

    def flatten(self):
        return [v.to_dict() for v in self.visits]


class ClinicalDataGenerator:
    def __init__(
        self,
        n_state_features: int = 30,
        n_intervention_features: int = 20,
        n_diagnoses: int = 10,
        n_clinical_patterns: int = 10,
        max_visits: int = 20,
    ):
        self.n_state = n_state_features
        self.n_interv = n_intervention_features
        self.n_diag = n_diagnoses
        self.n_patterns = n_clinical_patterns
        self.max_visits = max_visits

        # Create clinical patterns
        self.create_clinical_patterns()

    def create_clinical_patterns(self):
        """Clinical progression patterns"""
        self.patterns = []

        for i in range(self.n_patterns):
            pattern = {
                "state_indicators": np.random.choice(
                    self.n_state, size=5, replace=False
                ),
                "state_values": np.random.randn(5),
                "typical_interventions": np.random.choice(
                    self.n_interv, size=3, replace=False
                ),
                "associated_diagnoses": np.random.choice(
                    self.n_diag, size=2, replace=False
                ),
                "note_templates": [f"Pattern {i} with severity level {{}}", "Noise 123", "This is not relevant 2468"],
                # Add progression parameters
                "progression_rate": np.random.uniform(
                    0.1, 0.5
                ),  # How fast condition progresses
                "intervention_effectiveness": np.random.uniform(
                    0.5, 0.9
                ),  # How well interventions work
                "baseline_return_time": np.random.randint(
                    1, 6
                ),  # Typical months between visits
            }
            self.patterns.append(pattern)

    def generate_sample(
        self, pattern, severity
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """Generate a single clinical case"""

        # Generate state vector
        clinical_state = np.array([np.nan] * self.n_state)  # baseline null
        clinical_state[pattern["state_indicators"]] = pattern["state_values"] * severity

        # Generate interventions based on severity
        interventions = np.zeros(self.n_interv)
        if severity > 1.0:
            interventions[pattern["typical_interventions"]] = 1

        # Generate diagnoses
        diagnoses = np.zeros(self.n_diag)
        diagnoses[pattern["associated_diagnoses"]] = 1

        # Generate note
        note_lines = []
        for template in pattern["note_templates"]:
            note_lines.append(template.format(f"{severity:.1f}"))

        # Add some noise/variation
        if severity > 1.5:
            note_lines.append("Condition appears severe")
        elif severity < 0.7:
            note_lines.append("Mild presentation")

        return clinical_state, interventions, diagnoses, "\n".join(note_lines)

    def generate_patient_trajectory(self, patient_id: int) -> PatientTrajectory:
        """Generate complete patient trajectory"""

        # Select primary pattern for this patient
        pattern = np.random.choice(self.patterns)

        # Initialize patient state
        base_severity = np.random.uniform(0.5, 2.0)
        current_severity = base_severity

        visits = []
        n_visits = np.random.randint(1, self.max_visits + 1)

        for visit_id in range(n_visits):
            # Generate current visit
            clinical_state, interventions, diagnoses, note = self.generate_sample(
                pattern, current_severity
            )

            # Determine time to next visit based on severity and interventions
            if visit_id == n_visits - 1:
                # Last visit
                time_to_next = 13  # No more visits in 12 months
            else:
                # Calculate based on severity and intervention effectiveness
                intervention_effect = (
                    pattern["intervention_effectiveness"]
                    if interventions[pattern["typical_interventions"]].any()
                    else 0
                )

                base_time = pattern["baseline_return_time"]
                severity_modifier = max(0, 1 - current_severity)

                time_to_next = int(
                    base_time * (1 + severity_modifier + intervention_effect)
                )
                time_to_next = min(max(1, time_to_next), 12)

                # Small chance of death based on severity
                if current_severity > 1.8 and np.random.random() < 0.1:
                    time_to_next = 0
                    n_visits = visit_id + 1  # End trajectory

            visits.append(
                PatientVisit(
                    patient_id=patient_id,
                    visit_id=visit_id,
                    clinical_state=clinical_state,
                    interventions=interventions,
                    diagnoses=diagnoses,
                    note=note,
                    time_to_next=time_to_next,
                )
            )

            if time_to_next == 0:  # Death
                break

            # Progress disease for next visit
            progression = pattern["progression_rate"] * time_to_next
            intervention_effect = (
                pattern["intervention_effectiveness"]
                if interventions[pattern["typical_interventions"]].any()
                else 0
            )
            current_severity += progression * (1 - intervention_effect)

        return PatientTrajectory(patient_id=patient_id, visits=visits)

    def generate_dataset(self, n_patients: int) -> List[PatientTrajectory]:
        """Generate dataset of patient trajectories"""
        return list(itertools.chain(*[self.generate_patient_trajectory(i).flatten() for i in range(n_patients)]))