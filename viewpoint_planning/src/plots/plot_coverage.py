# plots/plot_coverage.py

import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')   # ← Pencere açmadan PNG kaydeder



COLORS = {
    'RH-NBV':   '#2ca02c',
    'Gradient': '#1f77b4',
    'PSO':      '#ff7f0e',
    'Random':   '#d62728',
}

MARKERS = {
    'RH-NBV':   'o',
    'Gradient': 's',
    'PSO':      '^',
    'Random':   'D',
}


def plot_coverage_progression(coverages, save_path=None, title='Coverage Progression'):
    """
    Plot coverage values from experiment.
    
    Args:
        coverages: dict of {method_name: numpy_array_from_experiment}
                   Bu array'leri SEN doldurmuyorsun — experiment çalıştığında
                   self.coverages_rh, self.coverages_pso vs. otomatik dolar
        save_path: optional PNG path
        title: figure title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, cov in coverages.items():
        cov = np.asarray(cov)
        iterations = np.arange(len(cov))

        ax.plot(
            iterations, cov,
            color=COLORS.get(method, 'black'),
            marker=MARKERS.get(method, 'o'),
            markersize=7,
            linewidth=2,
            label=method,
            alpha=0.9,
        )

        ax.annotate(
            f'{cov[-1]:.1f}%',
            xy=(iterations[-1], cov[-1]),
            xytext=(8, 0), textcoords='offset points',
            fontsize=10,
            color=COLORS.get(method, 'black'),
            fontweight='bold',
            va='center',
        )

    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('ROI Coverage (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_ylim(-5, 105)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, ax
