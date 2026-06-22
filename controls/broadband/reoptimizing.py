import torch
from typing import List, Optional, Tuple
from .base import BroadbandControl

class ReoptimizingBroadbandControl(BroadbandControl):
    """
    Широкополосный контроль с реоптимизацией толщин предыдущих слоёв.
    Параметры:
        reoptimize_from_layer: номер слоя (с 0), с которого начинается реоптимизация.
        reoptimize_window: сколько последних слоёв пересматривать (None = все).
    """
    def __init__(
        self,
        optical_model,
        wavelengths,
        sigma_r_percent,
        max_rate_deviation: float = 3.0,
        rate_optimization_max_iter: int = 50,
        time_optimization_max_iter: int = 50,
        stable_rate_threshold: float = 1e-4,
        enable_rate_stabilization: bool = False,
        reoptimize_from_layer: Optional[int] = None,
        reoptimize_window: Optional[int] = None,
    ):
        super().__init__(
            optical_model=optical_model,
            wavelengths=wavelengths,
            sigma_r_percent=sigma_r_percent,
            max_rate_deviation=max_rate_deviation,
            rate_optimization_max_iter=rate_optimization_max_iter,
            time_optimization_max_iter=time_optimization_max_iter,
            stable_rate_threshold=stable_rate_threshold,
            enable_rate_stabilization=enable_rate_stabilization,
        )
        self.reoptimize_from_layer = reoptimize_from_layer
        self.reoptimize_window = reoptimize_window
        self.ref_thicknesses = None  # актуальные толщины предыдущих слоёв

    def initialize(self, layer_index: int, target_thickness: float, nominal_rate: float):
        super().initialize(layer_index, target_thickness, nominal_rate)
        self.ref_thicknesses = None

    def _optimize_rate(
        self,
        times: torch.Tensor,
        signals: torch.Tensor,
        prev_thicknesses: torch.Tensor
    ) -> float:
        """
        M1 с реоптимизацией толщин.
        """
        device = times.device

        # Определяем индексы слоёв для реоптимизации
        if (self.reoptimize_from_layer is not None and
            self.layer_index >= self.reoptimize_from_layer):
            reopt_indices = self._get_reoptimize_indices(prev_thicknesses)
        else:
            reopt_indices = []

        # Инициализируем ref_thicknesses при первом шаге
        if self.ref_thicknesses is None:
            self.ref_thicknesses = prev_thicknesses.clone()

        # Параметры для оптимизации: скорость + толщины
        params = [torch.tensor(self.estimated_rate, dtype=torch.float64, device=device, requires_grad=True)]
        for idx in reopt_indices:
            params.append(
                torch.tensor(self.ref_thicknesses[idx].item(), dtype=torch.float64, device=device, requires_grad=True)
            )

        optimizer = torch.optim.LBFGS(
            params,
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

            # Заполняем предыдущие слои
            if len(reopt_indices) > 0:
                # Начинаем с ref_thicknesses
                d_matrix[:N_prev, :] = self.ref_thicknesses.unsqueeze(1)
                # Заменяем пересматриваемые слои на параметры
                for i, idx in enumerate(reopt_indices):
                    d_matrix[idx, :] = params[i + 1]
            else:
                d_matrix[:N_prev, :] = prev_thicknesses.unsqueeze(1)

            # Текущий слой
            d_matrix[N_prev, :] = params[0] * times

            pred = self.optical_model.get_spectrum(d_matrix, self.wavelengths)
            loss = torch.sum((pred - signals) ** 2)
            loss.backward()
            return loss

        optimizer.step(loss_fn)

        # Обновляем оценки
        new_rate = params[0].detach().item()
        if len(reopt_indices) > 0:
            for i, idx in enumerate(reopt_indices):
                self.ref_thicknesses[idx] = params[i + 1].detach().item()

        return new_rate

    def _get_reoptimize_indices(self, prev_thicknesses: torch.Tensor) -> List[int]:
        """Возвращает индексы слоёв, которые нужно пересматривать."""
        n_prev = len(prev_thicknesses)
        if self.reoptimize_window is not None:
            start_idx = max(0, n_prev - self.reoptimize_window)
            return list(range(start_idx, n_prev))
        else:
            return list(range(n_prev))

    def _optimize_delta(
        self,
        current_time: float,
        prev_thicknesses: torch.Tensor
    ) -> float:
        """
        M2 с использованием ref_thicknesses (если доступны).
        """
        # Используем ref_thicknesses для согласованности с M1
        if self.ref_thicknesses is not None and len(self.ref_thicknesses) == len(prev_thicknesses):
            prev_for_target = self.ref_thicknesses
        else:
            prev_for_target = prev_thicknesses

        device = prev_thicknesses.device
        target_d = torch.cat([prev_for_target, torch.tensor([self.target_thickness], device=device)])
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
            d_vec = torch.cat([prev_for_target, torch.tensor([d_est], device=device)])
            pred = self.optical_model.get_spectrum(d_vec.unsqueeze(1), self.wavelengths).squeeze(0)
            loss = torch.sum((pred - target_spectrum) ** 2)
            loss.backward()
            return loss

        optimizer.step(loss_fn)
        return delta.detach().item()