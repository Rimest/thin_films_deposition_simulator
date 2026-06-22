import torch
from thin_films_deposition_simulator import (
    OpticalMaterial, OpticalModel,
    RateNoiseModel, MeasurementModel,
    FullHistoryOptimizationControl,
    DepositionSimulator,
    plot_results, plot_spectra_evolution
)

SEED = 42

# 1. Материалы
n_odd = OpticalMaterial(2.35)
n_even = OpticalMaterial(1.45)
n_sub = OpticalMaterial(1.52)

# 2. Оптическая модель
optical_model = OpticalModel(
    n_odd=n_odd,
    n_even=n_even,
    n_substrate=n_sub,
    angle=0.0,
    polarization='S',
    substrate_type='semi-infinite',
    output_units='percent'
)

wavelengths = torch.linspace(400, 800, 200)

# 3. Шумы
rate_noise = RateNoiseModel(sigma_r_percent=10.0)
meas_noise = MeasurementModel(
    sigma_rand=0.5,
    sigma_fluc=0.2,
    sigma_syst=0.1,
    spec=False
)

# 4. Стратегия управления (без реоптимизации)
control = FullHistoryOptimizationControl(
    optical_model=optical_model,
    wavelengths=wavelengths,
    sigma_r_percent=10.0,
    enable_rate_stabilization=True,
    stable_rate_threshold=1e-4
)

# 5. Симулятор
sim = DepositionSimulator(
    optical_model=optical_model,
    control_strategy=control,
    rate_noise_model=rate_noise,
    measurement_noise_model=meas_noise,
    tau=1.0,
    sigma_add=2.0,
    save_spectra_interval=1,
    seed=SEED
)

# 6. Дизайн и запуск
design = torch.tensor([100.0, 150.0, 100.0, 200.0, 120.0])
nominal_rates = {'H': 0.5, 'L': 0.5}

results = sim.run(design, wavelengths, nominal_rates)

# 7. Визуализация
plot_results(results, wavelengths)
plot_spectra_evolution(results['spectra_history'], layer_index=2, chip_index=0,
                       wavelengths=wavelengths.numpy())

# Для анимации (в Jupyter):
# from IPython.display import HTML
# ani = animate_spectra_evolution(results['spectra_history'], layer_index=2)
# HTML(ani.to_jshtml())