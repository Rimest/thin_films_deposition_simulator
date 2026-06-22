import torch
from typing import Tuple, Optional

class RateNoiseModel:
    """Модель шума скорости напыления."""
    def __init__(self, sigma_r_percent: float):
        self.sigma_r = sigma_r_percent / 100.0

    def sample(self, nominal_rate: float, device=None) -> float:
        if device is None:
            device = torch.device('cpu')
        noise = torch.randn(1, device=device) * self.sigma_r * nominal_rate
        return nominal_rate + noise.item()


class MeasurementModel:
    """
    Модель измерений спектра пропускания.
    Шумы:
        - случайный (rand): независимый для каждой длины волны и каждого измерения.
        - флуктуационный (fluc): одинаковый для всех длин волн в одном измерении,
          но меняется от измерения к измерению.
        - систематический (syst): постоянный для всех измерений и длин волн.
    Опционально: свёртка с аппаратной функцией щели (гаусс).
    """
    def __init__(
        self,
        sigma_rand: float,
        sigma_fluc: float,
        sigma_syst: float,
        spec: bool = False,
        sgm: float = 1.0,
        n: int = 9
    ):
        self.sigma_rand = sigma_rand
        self.sigma_fluc = sigma_fluc
        self.sigma_syst = sigma_syst
        self.spec = spec
        self.sgm = sgm
        self.n = n

        # Систематическая ошибка генерируется один раз (для всех измерений и длин волн)
        # Инициализируем None, чтобы сгенерировать при первом вызове
        self._syst_noise = None

    def apply(
        self,
        signal: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor or Tuple[torch.Tensor, dict]:
        device = signal.device
        dtype = signal.dtype
        N_wl = len(signal)

        # 1. Генерация случайного шума (зависит от длины волны и момента времени)
        rand_noise = torch.randn(N_wl, device=device, dtype=dtype) * self.sigma_rand

        # 2. Генерация флуктуационного шума (один скаляр на измерение, одинаков для всех длин волн)
        fluc_noise = torch.randn(1, device=device, dtype=dtype) * self.sigma_fluc
        fluc_noise = fluc_noise.expand(N_wl)  # расширяем до размерности сигнала

        # 3. Систематический шум (генерируется один раз)
        if self._syst_noise is None:
            # Если sigma_syst == 0, можно сразу заполнить нулями, но оставим генерацию
            self._syst_noise = torch.randn(N_wl, device=device, dtype=dtype) * self.sigma_syst
        syst_noise = self._syst_noise

        # 4. Учёт щели (свёртка с гауссом) – применяется к сигналу ДО добавления шума
        if self.spec:
            nn = (self.n - 1) // 2
            x = torch.arange(-nn, nn+1, device=device, dtype=dtype)
            kernel = torch.exp(-x**2 / (2 * self.sgm**2))
            kernel = kernel / kernel.sum()
            signal_smooth = torch.zeros_like(signal)
            for i in range(N_wl):
                start = max(0, i - nn)
                end = min(N_wl, i + nn + 1)
                k_start = nn - (i - start) if i - start < nn else 0
                k_end = nn + (end - i) if end - i <= nn else self.n
                weights = kernel[k_start:k_end]
                weights = weights / weights.sum()
                signal_smooth[i] = torch.dot(signal[start:end], weights)
            signal = signal_smooth

        # 5. Добавляем все шумы
        signal_noisy = signal + rand_noise + fluc_noise + syst_noise

        if return_components:
            components = {
                'rand': rand_noise,
                'fluc': fluc_noise,
                'syst': syst_noise
            }
            return signal_noisy, components
        else:
            return signal_noisy

    def reset_systematic_noise(self):
        """Сброс систематической ошибки (для нового эксперимента)."""
        self._syst_noise = None