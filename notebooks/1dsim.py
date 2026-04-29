import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from tqdm.notebook import tqdm
from scipy.interpolate import interp1d
import warnings

class CKPZTransport1D:
    """
    Псевдоспектральный решатель 1D консервативного KPZ (CKPZ)
    с переносом частиц в поле тяжести.
    """
    def __init__(self, L=1.0, nx=256,
                 nu=1.0, lam=1.0, D=0.1,
                 kappa=0.01, gamma=0.1,
                 N_particles=2000, seed=42):
        self.L = L
        self.nx = nx
        self.dx = L / self.nx

        self.nu = nu
        self.lam = lam
        self.D = D
        self.kappa = kappa
        self.gamma = gamma
        self.N = N_particles

        self._compute_wave_numbers()
        self.dt = 0.5 * self.dx**2 / (self.nu * np.max(self.K2) + 1e-15)
        self._update_operators()

        self.rng = np.random.default_rng(seed)
        self._init_state()

    def _compute_wave_numbers(self):
        k = 2.0 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)   # (nx,)
        self.K = k
        self.K2 = k**2
        self.K4 = k**4
        self.IK = 1j * k

    def _update_operators(self):
        nuK4 = self.nu * self.K4
        mask = nuK4 > 1e-15
        self.Lin_fac = np.exp(-nuK4 * self.dt)
        self.Nl_fac = np.where(mask,
                               (1.0 - np.exp(-nuK4 * self.dt)) / nuK4,
                               self.dt)
        self.noise_fac = np.where(mask,
                                  np.sqrt(self.D * self.K2 * (1.0 - np.exp(-2.0 * nuK4 * self.dt)) / (2.0 * nuK4)),
                                  0.0)
        self.noise_fac[0] = 0.0   # нулевая мода

    def _init_state(self):
        self.h = 0.01 * self.rng.normal(size=self.nx)
        self.h_hat = np.fft.fft(self.h)

        self.x_p = self.rng.uniform(0, self.L, self.N)
        self.x0 = self.x_p.copy()
        self.x_unwrap = self.x_p.copy()

    def _grad_interp(self, field):
        """Линейная интерполяция градиента ∂_x h в точки x_p."""
        # Центральные разности на периодической сетке
        fpad = np.pad(field, 1, mode='wrap')
        dfdx = (fpad[2:] - fpad[:-2]) / (2 * self.dx)   # (nx,)

        x_grid = np.linspace(0, self.L, self.nx, endpoint=False)
        # Используем линейную интерполяцию
        interp = interp1d(x_grid, dfdx, kind='linear',
                          bounds_error=False, fill_value=(dfdx[0], dfdx[-1]))
        # Точки для частиц с учётом периодичности
        x_p_mod = self.x_p % self.L
        return interp(x_p_mod)

    def _step(self):
        """Псевдоспектральный шаг ETD1."""
        h_hat = self.h_hat

        # Нелинейность: N = (λ/2) * k^2 * F[(∂_x h)^2]
        dhdx = np.fft.ifft(self.IK * h_hat).real
        grad_sq = dhdx**2
        N_hat = (0.5 * self.lam) * self.K2 * np.fft.fft(grad_sq)

        # Шум
        xi = self.rng.normal(size=self.nx)
        xi_hat = np.fft.fft(xi)

        h_hat = (self.Lin_fac * h_hat
                 + self.Nl_fac * N_hat
                 + self.noise_fac * xi_hat)
        self.h_hat = h_hat
        self.h = np.fft.ifft(h_hat).real

        # Частицы
        grad_x = self._grad_interp(self.h)
        dW = np.sqrt(2 * self.kappa * self.dt) * self.rng.normal(size=self.N)
        dx = -self.gamma * grad_x * self.dt + dW

        self.x_p = (self.x_p + dx) % self.L
        self.x_unwrap += dx

    def run_single(self, T_steps, record_every=50, max_frames=80,
                   dt=None, max_time=None, verbose=True):
        if dt is not None:
            self.dt = dt
        elif max_time is not None:
            self.dt = max_time / T_steps

        self._update_operators()

        frames_h = []
        frames_p = []
        W_vals, msd_vals, t_vals = [], [], []

        save_every = max(1, T_steps // max_frames) if max_frames else T_steps + 1
        next_frame = 0

        loop = tqdm(range(T_steps), desc="Интегрирование 1D") if verbose else range(T_steps)
        for n in loop:
            self._step()

            if n % record_every == 0:
                W_vals.append(np.std(self.h))
                msd = np.mean((self.x_unwrap - self.x0)**2)
                msd_vals.append(msd)
                t_vals.append(n * self.dt)

            if n == next_frame and len(frames_h) < max_frames:
                frames_h.append(self.h.copy())
                frames_p.append(self.x_p.copy())
                next_frame += save_every

        if not frames_h or (T_steps-1) != next_frame - save_every:
            frames_h.append(self.h.copy())
            frames_p.append(self.x_p.copy())

        return {
            'W': np.array(W_vals),
            'msd': np.array(msd_vals),
            'times': np.array(t_vals),
            'frames_h': frames_h,
            'frames_p': frames_p,
            'dt_used': self.dt
        }

    def run_ensemble(self, n_runs, T_steps, record_every=50,
                     dt=None, max_time=None, verbose=True):
        all_W, all_msd = [], []
        for run_idx in tqdm(range(n_runs), desc="Ансамбль 1D"):
            self.rng = np.random.default_rng(42 + run_idx * 1000)
            self._init_state()
            res = self.run_single(T_steps, record_every=record_every,
                                  max_frames=0, dt=dt, max_time=max_time,
                                  verbose=False)
            all_W.append(res['W'])
            all_msd.append(res['msd'])

        times = res['times']
        W_mean = np.mean(all_W, axis=0)
        W_std  = np.std(all_W, axis=0) / np.sqrt(n_runs)
        msd_mean = np.mean(all_msd, axis=0)
        msd_std  = np.std(all_msd, axis=0) / np.sqrt(n_runs)
        return {
            'times': times,
            'W_mean': W_mean, 'W_std': W_std,
            'msd_mean': msd_mean, 'msd_std': msd_std,
            'dt_used': res['dt_used']
        }

    def run_varying_L_ensemble(self, L_list, n_runs, T_steps,
                               record_every=50, max_time=1.0, verbose=True):
        W_means, W_stds = [], []
        times_common = None
        for L_val in tqdm(L_list, desc="Вариация L (1D)"):
            model = CKPZTransport1D(L=L_val, nx=self.nx, nu=self.nu, lam=self.lam,
                                    D=self.D, kappa=self.kappa, gamma=self.gamma,
                                    N_particles=self.N)
            ens = model.run_ensemble(n_runs, T_steps, record_every=record_every,
                                     max_time=max_time, verbose=False)
            W_means.append(ens['W_mean'])
            W_stds.append(ens['W_std'])
            if times_common is None:
                times_common = ens['times']
        return {
            'L': np.array(L_list),
            'times': times_common,
            'W_mean_list': W_means,
            'W_std_list': W_stds
        }

    def animate(self, results, cmap='plasma', interval=80):
        frames_h = results['frames_h']
        frames_p = results['frames_p']
        if not frames_h:
            raise ValueError("Нет кадров для анимации")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        x_grid = np.linspace(0, self.L, self.nx, endpoint=False)

        line, = ax1.plot(x_grid, frames_h[0])
        ax1.set_ylim(np.min(frames_h[0])-0.1, np.max(frames_h[0])+0.1)
        ax1.set_ylabel('h(x)')
        ax1.set_title('Поле высот')

        # Гистограмма плотности частиц
        hist_bins = 50
        xp0 = frames_p[0]
        counts, bins, patches = ax2.hist(xp0, bins=hist_bins, range=(0, self.L),
                                        density=True, color='gray', alpha=0.7)
        ax2.set_xlabel('x')
        ax2.set_ylabel('Плотность')
        ax2.set_title('Пространственное распределение частиц')

        def update(frame_idx):
            line.set_ydata(frames_h[frame_idx])
            ax1.set_ylim(np.min(frames_h[frame_idx])-0.1, np.max(frames_h[frame_idx])+0.1)
            ax2.clear()
            xp = frames_p[frame_idx]
            ax2.hist(xp, bins=hist_bins, range=(0, self.L), density=True, color='gray', alpha=0.7)
            ax2.set_xlabel('x')
            ax2.set_ylabel('Плотность')
            ax2.set_title('Пространственное распределение частиц')
            return [line]

        ani = FuncAnimation(fig, update, frames=len(frames_h),
                            interval=interval, blit=False)
        plt.tight_layout()
        return ani

    def plot_observables(self, data, loglog=False, ax=None):
        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        else:
            axes = ax

        times = data['times'] if 'times' in data else data['times']
        if 'W_mean' in data:
            Wm, Ws = data['W_mean'], data['W_std']
            msdm, msds = data['msd_mean'], data['msd_std']
            ax1, ax2 = axes
            if loglog:
                ax1.errorbar(times, Wm, yerr=Ws, fmt='b.', capsize=2)
                ax1.set_xscale('log'); ax1.set_yscale('log')
                ax2.errorbar(times, msdm, yerr=msds, fmt='r.', capsize=2)
                ax2.set_xscale('log'); ax2.set_yscale('log')
            else:
                ax1.errorbar(times, Wm, yerr=Ws, fmt='b.', capsize=2)
                ax2.errorbar(times, msdm, yerr=msds, fmt='r.', capsize=2)
        else:
            ax1, ax2 = axes
            if loglog:
                ax1.loglog(times, data['W'], 'b.')
                ax2.loglog(times, data['msd'], 'r.')
            else:
                ax1.plot(times, data['W'], 'b.')
                ax2.plot(times, data['msd'], 'r.')

        ax1.set_xlabel('t'); ax1.set_ylabel('W(t)'); ax1.grid(True)
        ax2.set_xlabel('t'); ax2.set_ylabel(r'$\langle \Delta x^2 \rangle$'); ax2.grid(True)
        return axes

    def plot_scaling_collapse(self, scaling_data, a=0.5, b=0.3, loglog=True):
        L_vals = scaling_data['L']
        times = scaling_data['times']
        plt.figure(figsize=(8, 6))
        for i, L in enumerate(L_vals):
            Wm = scaling_data['W_mean_list'][i]
            scaled_W = Wm / (L**a)
            scaled_t = times / (L**b)
            if loglog:
                plt.loglog(scaled_t, scaled_W, label=f'L={L}')
            else:
                plt.plot(scaled_t, scaled_W, label=f'L={L}')
        plt.xlabel(f'$t / L^{b}$')
        plt.ylabel(f'$W / L^{a}$')
        plt.title(f'Скейлинг: a={a}, b={b}')
        plt.legend(); plt.grid(True)
        plt.show()


# Пример использования
if __name__ == '__main__':
    model = CKPZTransport1D(L=1.0, nx=512, nu=1.0, lam=1.0, D=0.1,
                            kappa=0.01, gamma=0.1, N_particles=2000)

    # Одиночный прогон
    res = model.run_single(T_steps=2000, record_every=20, max_frames=60, max_time=2.0)

    # Анимация
    ani = model.animate(res, interval=80)
    HTML(ani.to_jshtml())

    # Графики наблюдаемых
    model.plot_observables(res, loglog=True)
    plt.show()

    # Ансамбль
    ens = model.run_ensemble(n_runs=10, T_steps=1000, record_every=20, max_time=2.0)
    model.plot_observables(ens, loglog=True)
    plt.show()

    # Скейлинг по L (осторожно, занимает время)
    L_list = [1.0, 2.0, 4.0]
    sc_data = model.run_varying_L_ensemble(L_list, n_runs=5, T_steps=500,
                                           record_every=50, max_time=1.0)
    model.plot_scaling_collapse(sc_data, a=0.5, b=0.3, loglog=True)

    # Супердиффузионный режим наблюдается при параметрах:
    # model = CKPZTransport1D(L=2.5, nx=512, nu=0.25, lam=1.0, D=1.0,
    #                        kappa=0.01, gamma=7.5, N_particles=2000)
    # res = model.run_single(T_steps=250000, record_every=50, max_frames=60, max_time=500.0)
