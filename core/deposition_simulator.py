import torch
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from .optical_model import OpticalModel
from .control_strategy import ControlStrategy
from .noise_models import RateNoiseModel, MeasurementModel


class DepositionSimulator:
    def __init__(
        self,
        optical_model: OpticalModel,
        control_strategy: ControlStrategy,
        rate_noise_model: RateNoiseModel,
        measurement_noise_model: MeasurementModel,
        tau: float = 1.0,
        sigma_add: float = 0.0,
        chip_groups: Optional[List[List[int]]] = None,
        save_spectra_interval: int = 1,
        seed: Optional[int] = None,
        dtype=torch.float64
    ):
        self.optical_model = optical_model
        self.control_strategy = control_strategy
        self.rate_noise = rate_noise_model
        self.meas_noise = measurement_noise_model
        self.tau = tau
        self.sigma_add = sigma_add
        self.chip_groups = chip_groups if chip_groups is not None else [list(range(1))]
        self.save_spectra_interval = save_spectra_interval
        self.seed = seed
        self.dtype = dtype
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.design_thicknesses = None
        self.n_layers = 0
        self.n_chips = len(self.chip_groups)
        self.actual_thicknesses = None
        self.estimated_thicknesses = None
        self.time_history = []
        self.signal_history = []
        self.spectra_history = []

    def run(
        self,
        design_thicknesses: Union[List[float], torch.Tensor],
        wavelengths: torch.Tensor,
        nominal_rates: Dict[str, float],
        external_measurements: Optional[List[List[torch.Tensor]]] = None
    ) -> Dict:
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        self.design_thicknesses = torch.tensor(design_thicknesses, dtype=self.dtype, device=self.device)
        self.n_layers = len(self.design_thicknesses)
        self.wavelengths = wavelengths.to(self.device)

        n_chips = self.n_chips
        self.actual_thicknesses = torch.zeros((n_chips, self.n_layers), dtype=self.dtype, device=self.device)
        self.estimated_thicknesses = torch.zeros((n_chips, self.n_layers), dtype=self.dtype, device=self.device)
        self.time_history = [[] for _ in range(self.n_layers)]
        self.signal_history = [[] for _ in range(self.n_layers)]
        self.spectra_history = [[] for _ in range(self.n_layers)]

        for i in range(self.n_layers):
            layer_target = self.design_thicknesses[i]
            material = 'H' if (i % 2 == 0) else 'L'
            nominal_rate = nominal_rates[material]

            self.control_strategy.initialize(i, layer_target, nominal_rate)
            current_thicknesses = self.actual_thicknesses[:, :i]

            for ch in range(n_chips):
                prev_thick = current_thicknesses[ch, :] if i > 0 else torch.tensor([], device=self.device)
                self._deposit_layer(
                    chip_index=ch,
                    layer_index=i,
                    target=layer_target,
                    nominal_rate=nominal_rate,
                    prev_thicknesses=prev_thick,
                    external_measurements=external_measurements[i] if external_measurements is not None else None
                )

        self.error_history = self.actual_thicknesses - self.design_thicknesses
        return {
            'design': self.design_thicknesses,
            'actual': self.actual_thicknesses,
            'estimated': self.estimated_thicknesses,
            'errors': self.error_history,
            'time_history': self.time_history,
            'signal_history': self.signal_history,
            'spectra_history': self.spectra_history
        }

    def _deposit_layer(
        self,
        chip_index: int,
        layer_index: int,
        target: float,
        nominal_rate: float,
        prev_thicknesses: torch.Tensor,
        external_measurements: Optional[List[torch.Tensor]] = None
    ):
        t = 0.0
        d_actual = 0.0
        d_estimated = 0.0
        step_count = 0
        stop = False
        control = self.control_strategy
        layer_spectra = []

        while not stop:
            if external_measurements is not None and step_count < len(external_measurements):
                measured_signal = external_measurements[step_count]
                components = None
            else:
                full_thickness = torch.cat([prev_thicknesses, torch.tensor([d_actual], device=self.device)])
                theoretical_signal = self.optical_model.get_spectrum(
                    full_thickness.unsqueeze(1), self.wavelengths
                ).squeeze(0)
                measured_signal, components = self.meas_noise.apply(
                    theoretical_signal, return_components=True
                )

            if step_count % self.save_spectra_interval == 0:
                full_thickness = torch.cat([prev_thicknesses, torch.tensor([d_actual], device=self.device)])
                theo_spec = self.optical_model.get_spectrum(
                    full_thickness.unsqueeze(1), self.wavelengths
                ).squeeze(0).detach().cpu().numpy()
                meas_spec = measured_signal.detach().cpu().numpy()

                entry = {
                    'time': t,
                    'theoretical': theo_spec,
                    'measured': meas_spec,
                    'actual_thickness': d_actual,
                    'estimated_thickness': d_estimated,
                    'chip': chip_index
                }
                if components is not None:
                    entry['noise_rand'] = components['rand'].detach().cpu().numpy()
                    entry['noise_fluc'] = components['fluc'].detach().cpu().numpy()
                    entry['noise_syst'] = components['syst'].detach().cpu().numpy()
                layer_spectra.append(entry)

            decision = control.step(
                current_time=t,
                measured_signal=measured_signal,
                previous_layers_thicknesses=prev_thicknesses,
                tau=self.tau
            )

            delta_tau = decision.get('delta_tau', self.tau)
            stop = decision.get('stop', False)
            if stop:
                break

            rate_real = self.rate_noise.sample(nominal_rate, device=self.device)
            d_actual += rate_real * delta_tau
            t += delta_tau
            d_estimated = decision.get('estimated_thickness', nominal_rate * t)
            step_count += 1

        # Сохраняем историю для слоя
        self.spectra_history[layer_index].extend(layer_spectra)