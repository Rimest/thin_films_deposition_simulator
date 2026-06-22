import torch
from typing import Union, Callable, Tuple

class OpticalMaterial:
    """
    Оптический материал с дисперсией.
    Поддерживает:
        - постоянный n (и k=0)
        - таблицу [lambda, n] или [lambda, n, k] с интерполяцией
        - функцию, возвращающую (n, k) или комплексное n
    """
    def __init__(self, data: Union[float, torch.Tensor, Callable]):
        self.data = data
        self.type = None
        self.const_n = None
        self.table = None
        self.has_k = False
        self.func = None

        if isinstance(data, (int, float)):
            self.type = 'constant'
            self.const_n = float(data)
        elif isinstance(data, torch.Tensor):
            if data.ndim == 2 and data.shape[1] >= 2:
                self.type = 'table'
                self.table = data
                self.has_k = data.shape[1] >= 3
            else:
                raise ValueError("Тензор должен быть формы (N,2) или (N,3)")
        elif callable(data):
            self.type = 'function'
            self.func = data
        else:
            raise TypeError("Неподдерживаемый тип данных")

    def get_values(self, wavelengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Возвращает (n, k) для заданных длин волн.
        wavelengths: (N,) тензор
        Возвращает: (n, k) тензоры той же формы.
        """
        if self.type == 'constant':
            n = torch.full_like(wavelengths, self.const_n, dtype=torch.float64)
            k = torch.zeros_like(wavelengths, dtype=torch.float64)
            return n, k

        elif self.type == 'table':
            lam = self.table[:, 0].to(wavelengths.device)
            n_vals = self.table[:, 1].to(wavelengths.device)
            if self.has_k:
                k_vals = self.table[:, 2].to(wavelengths.device)
            else:
                k_vals = torch.zeros_like(lam)

            n = torch.interp(wavelengths, lam, n_vals)
            k = torch.interp(wavelengths, lam, k_vals)
            return n, k

        elif self.type == 'function':
            result = self.func(wavelengths)
            if isinstance(result, tuple) and len(result) == 2:
                n, k = result
            else:
                n = torch.real(result)
                k = torch.imag(result)
            if not torch.is_tensor(n):
                n = torch.tensor(n, dtype=torch.float64, device=wavelengths.device)
                k = torch.tensor(k, dtype=torch.float64, device=wavelengths.device)
            return n, k

        else:
            raise RuntimeError("Неизвестный тип материала")