from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import math
from typing import Tuple
import numpy as np

from genalgo.population import Population
from genalgo.crossover import sbx
from genalgo.selection import tournament_select
from genalgo.mutation import gaussian

import rovibrational_excitation as rve

# --- 1. Basis & dipole matrices ----------------------------------
c_vacuum = 299792458 * 1e2 / 1e15  # cm/fs
debye_unit = 3.33564e-30                       # 1 D → C·m
Omega01_rad_phz = 2349*2*np.pi*c_vacuum
Delta_omega_rad_phz = 25*2*np.pi*c_vacuum
B_rad_phz = 0.39e-3*2*np.pi*c_vacuum
Mu0_Cm = 0.3 * debye_unit                      # 0.3 Debye 相当
Potential_type = "morse"  # or "morse"
V_max = 4
J_max = 1

basis = rve.LinMolBasis(
            V_max=V_max,
            J_max=J_max,
            use_M = True,
            omega_rad_phz = Omega01_rad_phz,
            delta_omega_rad_phz = Delta_omega_rad_phz
            )           # |v J M⟩ direct-product

dip   = rve.LinMolDipoleMatrix(
            basis, mu0=Mu0_Cm, potential_type=Potential_type,
            backend="numpy", dense=True)            # CSR on GPU

mu_x  = dip.mu_x            # lazy-built, cached thereafter
mu_y  = dip.mu_y
mu_z  = dip.mu_z

# --- 2. Hamiltonian ----------------------------------------------
H0 = rve.generate_H0_LinMol(
        basis,
        omega_rad_phz       = Omega01_rad_phz,
        delta_omega_rad_phz = Delta_omega_rad_phz,
        B_rad_phz           = B_rad_phz,
)

# --- 3. Electric field -------------------------------------------
dt, te = 0.1, 100000
TIME  = np.arange(0, te+dt, dt)

# --- 4. Initial state |v=0,J=0,M=0⟩ ------------------------------
from rovibrational_excitation.core.states import StateVector
psi0 = StateVector(basis)
psi0.set_state((0,0,0), 1.0)
psi0.normalize()

# --- 5. Time propagation (Schrödinger) ---------------------------
def propagation(dispersion):
    E  = rve.ElectricField(tlist=TIME)

    E.add_dispersed_Efield(
            envelope_func=rve.core.electric_field.gaussian_fwhm,
            duration=100.0,             # FWHM (fs)
            t_center=5000,
            carrier_freq=2349*c_vacuum,   # rad/fs
            amplitude=1.0e12,
            polarization=[1.0, 0.0],
            gdd = dispersion[0],
            tod = dispersion[1]
    )
    psi = rve.schrodinger_propagation(
                H0, E, dip,
                psi0.data,
                axes="zx",
                return_traj=False,
                backend="numpy",
                )
    return psi[0]

def fitness_fn(dispersion: np.ndarray) -> float:
    """Scalar fitness = sum of squares → 0 when both equations are satisfied."""
    psi = propagation(dispersion)
    return 1 / (1 + np.abs(psi[16])**2)


# ------------------------------------------------------------
# GA CONFIG — tweak these values only
# ------------------------------------------------------------
POP_SIZE = 10       # 集団サイズ
GENERATIONS = 10    # 世代数
SEED = 0             # 乱数シード
BOUNDS = ((-3.0e4, 3e4), (-1, 1))  # 探索範囲 (x_min, x_max), (y_min, y_max)
MUTATION_SIGMA = 0.2 # ガウス変異の σ

# ------------------------------------------------------------
# GA execution
# ------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(SEED)
    init_genes = rng.uniform(
        [b[0] for b in BOUNDS], [b[1] for b in BOUNDS], size=(POP_SIZE, 2)
    )

    pop = Population(init_genes, fitness_fn, rng=rng)

    best_gene, best_fit = pop.evolve(
        generations=GENERATIONS,
        selector=lambda f, r: tournament_select(f, k=3, rng=r),
        crossover_op=lambda a, b, r: sbx(a, b, eta=1.0, rng=r),
        mutation_op=lambda x, r: gaussian(
            x, sigma=MUTATION_SIGMA, prob=1.0, bounds=BOUNDS, rng=r
        ),
        bounds=BOUNDS,
        verbose=True,
    )

    gdd, tod = best_gene
    
    print("\nBest candidate found:")
    print(f"gdd = {gdd:.6f}, tod = {tod:.6f}")
    print(f"sum of squares = {best_fit:.3e}")
    print(f"population = {1/best_fit-1:.3e}")


if __name__ == "__main__":
    main()
