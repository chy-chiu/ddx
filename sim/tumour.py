import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Literal
import torch

REMISSION_THRESHOLD = 0.1
CHEMO_DOSAGE = 5
RT_DOSAGE = 2

@dataclass
class TumourSimParameters:
    rho: float = 7e-5  # growth rate
    initial_d: float = 1.72 # Initial diameter
    K: float = 30 ** 3 * np.pi / 6  # carrying capacity
    beta_c: float = 0.028  # chemo effect
    alpha_r: float = 0.0398  # radio effect linear
    ab_ratio: float = 10  # alpha-beta ratio of radio effect (linear-quadratic ratio)
    gamma: float = 1.0  # confounding strength
    d_max: float = 13.0  # max tumor diameter
    noise_std: float = 1e-4  # std of noise term
    regime: Literal[1, 2, 3] = 0  # clinical regimen arm
    treatment_delay: int = 10

class TumourSimulator:
    """Extended simulator of lung cancer patient trajectory using the Geng model, featuring
    clinical notes and additional symptoms etc.
    """

    def __init__(self, params: TumourSimParameters = None):
        self.params = params or TumourSimParameters()

    def _compute_chemo_effect(self, t: int, treatment_times: List[int]) -> float:
        """Compute exponentially decaying chemotherapy effect"""
        if not treatment_times:
            return 0.0

        # Sum effect from all past treatments
        effect = np.sum(np.array([0.5]) ** (t - np.array(treatment_times))) * CHEMO_DOSAGE
        return self.params.beta_c * effect

    def _assign_treatment(self, volume_traj: List[float]) -> Tuple[bool, bool]:
        """Assign treatments based on previous tumor volume"""
        
        if len(volume_traj) < self.params.treatment_delay:
            return False, False
        
        # Convert volume to diameter (assuming spherical tumor)
        
        volume_traj = np.array(volume_traj[-15:]) if len(volume_traj) > 15 else np.array(volume_traj)
        
        diameter_traj =(6 * volume_traj / np.pi) ** (1/3)
        d_15 = np.mean(diameter_traj)

        # Compute probability
        prob = torch.Tensor([-self.params.gamma * (d_15 - self.params.d_max / 2) / self.params.d_max])
        prob = torch.sigmoid(prob)
        
        # Assign treatments
        chemo = np.random.random() < prob
        radio = np.random.random() < prob

        return chemo, radio

    def _evolve_tumor(
        self, Y_t: float, chemo_effect: float, radio_effect: float
    ) -> float:
        """Evolve tumor volume one step"""
        growth_term = 1 + self.params.rho * np.log(self.params.K / Y_t)
        treatment_effect = chemo_effect + radio_effect
        noise = np.random.normal(0, self.params.noise_std)

        Y_next = Y_t * (growth_term - treatment_effect + noise)
        return max(Y_next, REMISSION_THRESHOLD)  # ensure volume stays positive

    def generate_clinical_note(
        self, t: int, Y_curr: float, Y_prev: float, chemo_given: bool, radio_given: bool
    ) -> str:
        """Generate synthetic clinical note"""

        # Compute relative change
        rel_change = (Y_curr - Y_prev) / Y_prev

        # Status description
        if rel_change > 0.05:
            status = "shows concerning progression"
            concern = "high"
        elif rel_change < -0.05:
            status = "demonstrates therapeutic response"
            concern = "moderate"
        else:
            status = "appears stable"
            concern = "moderate"

        # Treatment description
        treatments = []
        if chemo_given:
            treatments.append("Chemotherapy cycle administered")
        if radio_given:
            treatments.append("Radiation therapy delivered")
        treatment_text = (
            ". ".join(treatments) if treatments else "No treatment given today"
        )

        note = f"""
            Day {t} Clinical Note
            -------------------
            Tumor Assessment:
            Tumor volume measured at {Y_curr:.1f}cc ({rel_change*100:.1f}% change). Lesion {status}.

            Treatment Summary:
            {treatment_text}.

            Clinical Assessment:
            Patient continues under {concern} monitoring. 
            {"Treatment response will be closely evaluated." if treatments else "Will continue current management plan."}
        """

        return note.strip()

    def simulate_trajectory(self, num_steps: int) -> Dict:
        """Simulate complete trajectory with clinical notes"""

        trajectory = {
            "time": [],
            "volume": [],
            "chemo_given": [],
            "radio_given": [],
            "notes": [],
        }

        Y_t = np.pi * (self.params.initial_d ** 3) / 6
        
        chemo_times = []
        
        Y_max = np.pi * (self.params.d_max ** 3) / 6
        
        radio_effect = self.params.alpha_r * RT_DOSAGE + (self.params.alpha_r / self.params.ab_ratio) * RT_DOSAGE ** 2
        
        for t in range(num_steps):
            # Assign treatments
            chemo, radio = self._assign_treatment(trajectory['volume'])

            # Compute treatment effects
            if chemo:
                chemo_times.append(t)
            chemo_effect = self._compute_chemo_effect(t, chemo_times)
            radio_effect = radio_effect if radio else 0

            # Evolve tumor
            Y_next = self._evolve_tumor(Y_t, chemo_effect, radio_effect)

            # Generate note
            note = self.generate_clinical_note(t, Y_next, Y_t, chemo, radio)

            # Store everything
            trajectory["time"].append(t)
            trajectory["volume"].append(Y_next)
            trajectory["chemo_given"].append(chemo)
            trajectory["radio_given"].append(radio)
            trajectory["notes"].append(note)

            Y_t = Y_next
            if Y_t > Y_max:
                break

        return trajectory


# Usage example:
if __name__ == "__main__":
    # Create simulator
    params = TumourSimParameters(initial_d=3.86, rho=0.00723)

    simulator = TumourSimulator(params=params)

    # Simulate trajectory
    trajectory = simulator.simulate_trajectory(num_steps=100)

    # Print example note
    # print(trajectory["notes"][0])
    print(trajectory['volume'])

    # You could also plot the trajectory:
    import matplotlib.pyplot as plt
    plt.plot(trajectory['time'], trajectory['volume'])
    plt.show()
