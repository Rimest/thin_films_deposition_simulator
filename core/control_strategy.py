from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch

class ControlStrategy(ABC):
    """
    Абстрактный базовый класс для всех стратегий управления напылением.
    """
    def __init__(self):
        self.layer_index = None
        self.target_thickness = None
        self.nominal_rate = None
        self.history = []  # можно хранить историю измерений, если нужно

    @abstractmethod
    def initialize(self, layer_index: int, target_thickness: float, nominal_rate: float):
        """
        Подготовка к напылению слоя.
        """
        self.layer_index = layer_index
        self.target_thickness = target_thickness
        self.nominal_rate = nominal_rate
        self.history = []

    @abstractmethod
    def step(
        self,
        current_time: float,
        measured_signal: torch.Tensor,
        previous_layers_thicknesses: torch.Tensor,
        tau: float,
        **kwargs
    ) -> Dict:
        """
        Вызывается на каждом временном шаге.
        Возвращает словарь с полями:
            'stop': bool – остановить слой?
            'delta_tau': float – длительность следующего шага (адаптивный шаг)
            'estimated_rate': float – оценка скорости (опционально)
            'estimated_thickness': float – оценка толщины (опционально)
        """
        pass