import torch
from abc import abstractmethod
from typing import Dict, List, Optional, Tuple
from ...core.control_strategy import ControlStrategy
from ...core.optical_model import OpticalModel


class BroadbandControl(ControlStrategy):
    """
    Базовый класс для широкополосного контроля с историей измерений.
    Содержит общие методы для M1, M2 и принятия решений.
    Подклассы должны реализовать _build_optimization_params и _compute_M1.
    """
    def __init__(
        self,
        optical_model: OpticalModel,
        wavelengths: torch.Tensor,
        sigma_r_percent: float,
        max_rate_deviation: float = 3.0,
        rate_optimization_max_iter: int = 50,
        time_optimization_max_iter: int = 50,
        stable_rate_threshold: float = 1e-4,
        enable_rate_stabilization: bool = False,
    ):
        super().__init__()
        self.optical_model = optical_model
        self.wavelengths = wavelengths
        self.sigma_r_percent = sigma_r_percent
        self.max_rate_deviation = max_rate_deviation
        self.rate_optimization_max_iter = rate_optimization_max_iter
        self.time_optimization_max_iter = time_optimization_max_iter
        self.stable_rate_threshold = stable_rate_threshold
        self.enable_rate_stabilization = enable_rate_stabilization

        self.measured_history = []  # список (time, signal)
        self.estimated_rate = None
        self.rate_stable = False
        self.prev_rate = None

    def initialize(self, layer_index: int, target_thickness: float, nominal_rate: float):
        super().initialize(layer_index, target_thickness, nominal_rate)
        self.measured_history = []
        self.estimated_rate = nominal_rate
        self.rate_stable = False
        self.prev_rate = None

    def step(
        self,
        current_time: float,
        measured_signal: torch.Tensor,
        previous_layers_thicknesses: torch.Tensor,
        tau: float,
        **kwargs
    ) -> Dict:
        device = measured_signal.device
        self.measured_history.append((current_time, measured_signal))

        # На первом шаге используем номинальную скорость
        if len(self.measured_history) == 1:
            self.estimated_rate = self.nominal_rate
        else:
            # M1: оценка скорости (если не стабилизирована)
            if not (self.rate_stable and self.enable_rate_stabilization):
                times, signals = self._prepare_history(device)
                rate = self._optimize_rate(times, signals, previous_layers_thicknesses)
                self.estimated_rate = self._apply_rate_constraint(rate)
                self._update_rate_stability(rate)

        # Страховка от перепыления
        estimated_thickness = self.estimated_rate * current_time
        if estimated_thickness > self.target_thickness + 1e-6:
            return self._make_stop_decision(estimated_thickness)

        # M2: оценка оставшегося времени
        delta_tau_est = self._optimize_delta(current_time, previous_layers_thicknesses)
        stop, delta_tau = self._make_step_decision(delta_tau_est, tau)

        return {
            'stop': stop,
            'delta_tau': delta_tau,
            'estimated_rate': self.estimated_rate,
            'estimated_thickness': estimated_thickness,
            'delta_tau_est': delta_tau_est
        }

    # ---------- Вспомогательные методы ----------

    def _prepare_history(self, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Возвращает (times, signals) для всех сохранённых измерений."""
        times = torch.tensor([t for t, _ in self.measured_history], dtype=torch.float64, device=device)
        signals = torch.stack([s for _, s in self.measured_history])  # (N_meas, N_wl)
        return times, signals

    def _optimize_rate(
        self,
        times: torch.Tensor,
        signals: torch.Tensor,
        prev_thicknesses: torch.Tensor
    ) -> float:
        """
        M1: минимизация невязки по скорости (без реоптимизации толщин).
        Возвращает оптимальную скорость.
        """
        device = times.device
        rate = torch.tensor(self.estimated_rate, dtype=torch.float64, device=device, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [rate],
            lr=0.1,
            max_iter=self.rate_optimization_max_iter,
            history_size=10,
            line_search_fn='strong_wolfe'
        )

        N_prev = len(prev_thicknesses)
        N_meas = len(times)

        def loss_fn():
            optimizer.zero_grad()
            d_matrix = torch.zeros((N_prev + 1, N_meas), dtype=torch.float64, device=device)
            d_matrix[:N_prev, :] = prev_thicknesses.unsqueeze(1)
            d_matrix[N_prev, :] = rate * times
            pred = self.optical_model.get_spectrum(d_matrix, self.wavelengths)  # (N_meas, N_wl)
            loss = torch.sum((pred - signals) ** 2)
            loss.backward()
            return loss

        optimizer.step(loss_fn)
        return rate.detach().item()

    def _optimize_delta(
        self,
        current_time: float,
        prev_thicknesses: torch.Tensor
    ) -> float:
        """
        M2: оценка оставшегося времени до целевой толщины.
        Возвращает оптимальное delta_tau.
        """
        device = prev_thicknesses.device
        # Целевой спектр
        target_d = torch.cat([prev_thicknesses, torch.tensor([self.target_thickness], device=device)])
        target_spectrum = self.optical_model.get_spectrum(target_d.unsqueeze(1), self.wavelengths).squeeze(0)

        delta = torch.tensor(0.0, dtype=torch.float64, device=device, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [delta],
            lr=0.1,
            max_iter=self.time_optimization_max_iter,
            history_size=10,
            line_search_fn='strong_wolfe'
        )

        def loss_fn():
            optimizer.zero_grad()
            d_est = self.estimated_rate * (current_time + delta)
            d_vec = torch.cat([prev_thicknesses, torch.tensor([d_est], device=device)])
            pred = self.optical_model.get_spectrum(d_vec.unsqueeze(1), self.wavelengths).squeeze(0)
            loss = torch.sum((pred - target_spectrum) ** 2)
            loss.backward()
            return loss

        optimizer.step(loss_fn)
        return delta.detach().item()

    def _apply_rate_constraint(self, rate: float) -> float:
        """Ограничивает скорость в пределах [r_min, r_max]."""
        r_min = self.nominal_rate * (1 - self.max_rate_deviation * self.sigma_r_percent / 100.0)
        r_max = self.nominal_rate * (1 + self.max_rate_deviation * self.sigma_r_percent / 100.0)
        return max(r_min, min(rate, r_max))

    def _update_rate_stability(self, new_rate: float):
        """Обновляет флаг стабилизации скорости."""
        if self.enable_rate_stabilization and self.prev_rate is not None:
            rel_change = abs((new_rate - self.prev_rate) / (self.prev_rate + 1e-12))
            self.rate_stable = rel_change < self.stable_rate_threshold
        self.prev_rate = new_rate

    def _make_step_decision(self, delta_tau_est: float, tau: float) -> Tuple[bool, float]:
        """Принимает решение о шаге и остановке."""
        if delta_tau_est < 0:
            return True, 0.0
        elif delta_tau_est < tau:
            return True, delta_tau_est
        else:
            return False, tau

    def _make_stop_decision(self, estimated_thickness: float) -> Dict:
        """Возвращает решение об экстренной остановке."""
        return {
            'stop': True,
            'delta_tau': 0.0,
            'estimated_rate': self.estimated_rate,
            'estimated_thickness': estimated_thickness,
            'delta_tau_est': 0.0
        }