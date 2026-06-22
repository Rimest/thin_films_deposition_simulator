import torch
from typing import Optional, Union, Tuple
from .material import OpticalMaterial

class OpticalModel:
    """
    Модель для расчёта спектров пропускания многослойных покрытий.
    Использует матричный метод (характеристические матрицы).
    Поддерживает пакетный режим: thicknesses имеет форму (N_layers, batch_size).
    """
    def __init__(
        self,
        n_odd: OpticalMaterial,
        n_even: OpticalMaterial,
        n_substrate: OpticalMaterial,
        angle: float,
        polarization: str,
        substrate_thickness: float = 0.0,
        substrate_type: str = 'semi-infinite',
        output_units: str = 'percent',
        dtype=torch.float64
    ):
        self.n_odd = n_odd
        self.n_even = n_even
        self.n_sub = n_substrate
        self.angle = angle
        self.polarization = polarization.upper()
        self.substrate_thickness = substrate_thickness
        self.substrate_type = substrate_type
        self.output_units = output_units
        self.dtype = dtype

    def get_spectrum(
        self,
        thicknesses: torch.Tensor,   # (N_layers, batch_size) или (batch_size, N_layers)
        wavelengths: torch.Tensor    # (N_wl,)
    ) -> torch.Tensor:
        """
        Возвращает спектр пропускания для каждого набора толщин.
        Выход: (batch_size, N_wl) – коэффициенты в процентах или долях.
        """
        # Приведение к форме (N_layers, batch_size)
        if thicknesses.ndim == 1:
            thicknesses = thicknesses.unsqueeze(1)
        elif thicknesses.ndim == 2 and thicknesses.shape[1] > 1 and thicknesses.shape[0] != 1:
            thicknesses = thicknesses.T
        elif thicknesses.ndim == 2 and thicknesses.shape[1] == 1:
            pass
        else:
            raise ValueError("Некорректная форма thicknesses")

        return self._compute_spectrum_batch(thicknesses, wavelengths)

    def _compute_spectrum_batch(
        self,
        d: torch.Tensor,            # (N_layers, batch_size)
        wavelengths: torch.Tensor   # (N_wl,)
    ) -> torch.Tensor:
        N_layers, batch_size = d.shape
        N_wl = len(wavelengths)
        device = d.device
        dtype = self.dtype

        # Угол падения
        theta_a = torch.deg2rad(torch.tensor(self.angle, dtype=dtype, device=device))
        sin_theta_a = torch.sin(theta_a)
        cos_theta_a = torch.cos(theta_a)
        na = torch.tensor(1.0, dtype=dtype, device=device)

        # Получаем n и k для слоёв и подложки
        n_odd_vals, _ = self.n_odd.get_values(wavelengths)
        n_even_vals, _ = self.n_even.get_values(wavelengths)
        n_sub_vals, k_sub_vals = self.n_sub.get_values(wavelengths)

        # Матрица n для слоёв: (N_layers, N_wl)
        n_layers = torch.zeros((N_layers, N_wl), dtype=dtype, device=device)
        for j in range(N_layers):
            if j % 2 == 0:
                n_layers[j, :] = n_odd_vals
            else:
                n_layers[j, :] = n_even_vals

        # Углы преломления в слоях
        sin_theta_i = (na / n_layers) * sin_theta_a
        sin_theta_i = torch.clamp(sin_theta_i, max=1.0)
        cos_theta_i = torch.sqrt(1.0 - sin_theta_i**2)

        # Угол в подложке
        sin_theta_s = (na / n_sub_vals) * sin_theta_a
        sin_theta_s = torch.clamp(sin_theta_s, max=1.0)
        cos_theta_s = torch.sqrt(1.0 - sin_theta_s**2)

        # Параметры q
        if self.polarization == 'S':
            q_layers = n_layers * cos_theta_i
            qa = na * cos_theta_a
            qs = n_sub_vals * cos_theta_s
        elif self.polarization == 'P':
            q_layers = n_layers / cos_theta_i
            qa = na / cos_theta_a
            qs = n_sub_vals / cos_theta_s
        else:
            raise ValueError("polarization must be 'S' or 'P'")

        k0 = 2 * torch.pi / wavelengths

        # Расширение размерностей для пакетного расчёта
        d_exp = d.unsqueeze(-1)                     # (N_layers, batch, 1)
        n_exp = n_layers.unsqueeze(1)               # (N_layers, 1, N_wl)
        cos_exp = cos_theta_i.unsqueeze(1)          # (N_layers, 1, N_wl)
        k0_exp = k0.unsqueeze(0).unsqueeze(0)       # (1, 1, N_wl)
        q_layers_exp = q_layers.unsqueeze(1)        # (N_layers, 1, N_wl)

        phi = k0_exp * n_exp * d_exp * cos_exp      # (N_layers, batch, N_wl)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)

        # Характеристическая матрица (batch, N_wl, 2, 2)
        M = torch.eye(2, dtype=torch.complex128, device=device).unsqueeze(0).unsqueeze(0)
        M = M.expand(batch_size, N_wl, 2, 2).contiguous()

        for j in range(N_layers):
            cos_j = cos_phi[j]          # (batch, N_wl)
            sin_j = sin_phi[j]
            q_j = q_layers_exp[j]       # (1, N_wl) -> транслируется

            M_j = torch.zeros((batch_size, N_wl, 2, 2), dtype=torch.complex128, device=device)
            M_j[:, :, 0, 0] = cos_j
            M_j[:, :, 0, 1] = 1j * sin_j / q_j
            M_j[:, :, 1, 0] = 1j * q_j * sin_j
            M_j[:, :, 1, 1] = cos_j
            M = torch.matmul(M, M_j)

        A, B, C, D = M[:, :, 0, 0], M[:, :, 0, 1], M[:, :, 1, 0], M[:, :, 1, 1]

        qa_exp = qa  # скаляр
        qs_exp = qs.unsqueeze(0).expand(batch_size, N_wl)

        denom = qa_exp * A + qa_exp * qs_exp * B + C + qs_exp * D
        r = (qa_exp * A + qa_exp * qs_exp * B - C - qs_exp * D) / denom
        t = 2 * qa_exp / denom

        R1 = torch.abs(r)**2
        T1 = torch.real(qs_exp / qa_exp) * torch.abs(t)**2

        R2 = ((qa_exp - qs_exp) / (qa_exp + qs_exp))**2
        T2 = 4 * qa_exp * qs_exp / ((qa_exp + qs_exp)**2)

        if self.substrate_type == 'semi-infinite':
            T = T1
        elif self.substrate_type == 'finite':
            alpha = 4 * torch.pi * k_sub_vals / wavelengths
            alpha_exp = alpha.unsqueeze(0).expand(batch_size, N_wl)
            Exp1 = torch.exp(-alpha_exp * self.substrate_thickness)
            Exp2 = torch.exp(-2 * alpha_exp * self.substrate_thickness)
            denom_fin = 1 - R1 * R2 * Exp2
            T = T1 * T2 * Exp1 / denom_fin
        else:
            raise ValueError("substrate_type must be 'semi-infinite' or 'finite'")

        if self.output_units == 'percent':
            T = 100 * T

        return T