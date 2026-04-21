import numpy as np

def laplacian(f, dx):
    """Вторая производная (Лапласиан) d^2f / dx^2"""
    return (np.roll(f, -1) - 2*f + np.roll(f, 1)) / (dx**2)

def gradient(f, dx):
    """Первая производная (градиент) df/dx"""
    return (np.roll(f, -1) - np.roll(f, 1)) / (2*dx)

def divergence(f_vector, dx):
    """Дивергенция (в 1D это просто df/dx)"""
    return (np.roll(f_vector, -1) - np.roll(f_vector, 1)) / (2*dx)

def get_noise(n, dx, dt, d_coeff):
    """Консервативный шум для CKPZ (уравнение 10)"""
    # Генерируем обычный белый шум
    white_noise = np.random.normal(0, 1, n)
    # По формуле <nn> = -D*d^2*delta, это значит,
    # что шум сам является второй производной
    noise = np.sqrt(d_coeff / (dx * dt)) * laplacian(white_noise, dx)
    return noise