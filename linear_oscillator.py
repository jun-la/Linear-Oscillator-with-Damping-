"""
Linear Oscillator with Damping Differential Equation Solver

This module solves the differential equation:
    m * d²x/dt² + c * dx/dt + k * x = 0

Where:
    m = mass
    c = damping coefficient
    k = spring constant
    x = displacement
    t = time
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class OscillatorParameters:
    """Parameters for the linear oscillator with damping."""
    mass: float
    damping_coefficient: float
    spring_constant: float
    initial_position: float
    initial_velocity: float


class LinearOscillatorSolver:
    """Solver for the linear oscillator with damping differential equation."""
    
    def __init__(self, params: OscillatorParameters):
        """
        Initialize the solver with oscillator parameters.
        
        Args:
            params: OscillatorParameters object containing mass, damping, spring constant, 
                   and initial conditions
        """
        self.params = params
        self._solution = None
        self._symbolic_solution = None
        
    def solve_symbolic(self) -> sp.Expr:
        """
        Solve the differential equation symbolically using SymPy.
        
        Returns:
            Symbolic solution as a SymPy expression
        """
        # Define symbols
        t = sp.Symbol('t', real=True)
        x = sp.Function('x')(t)
        
        # Define the differential equation: m*d²x/dt² + c*dx/dt + k*x = 0
        m, c, k = self.params.mass, self.params.damping_coefficient, self.params.spring_constant
        
        # Create the differential equation
        eq = m * sp.diff(x, t, 2) + c * sp.diff(x, t) + k * x
        
        # Solve the differential equation
        solution = sp.dsolve(eq, x)
        
        # Apply initial conditions
        x0, v0 = self.params.initial_position, self.params.initial_velocity
        
        # Get the general solution
        general_solution = solution.rhs
        
        # Find constants C1 and C2 from initial conditions
        # At t=0: x(0) = x0 and dx/dt(0) = v0
        t_val = 0
        
        # Substitute t=0 into the solution
        eq1 = general_solution.subs(t, t_val) - x0
        
        # Substitute t=0 into the derivative
        derivative = sp.diff(general_solution, t)
        eq2 = derivative.subs(t, t_val) - v0
        
        # Solve for constants
        constants = sp.solve([eq1, eq2], sp.symbols('C1 C2'))
        
        if constants:
            # Substitute the constants back into the solution
            self._symbolic_solution = general_solution.subs(constants)
        else:
            # If no constants found, use the general solution
            self._symbolic_solution = general_solution
            
        return self._symbolic_solution
    
    def solve_numerical(self, t_span: Tuple[float, float], num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the differential equation numerically and return time and position arrays.
        
        Args:
            t_span: Tuple of (start_time, end_time)
            num_points: Number of points to generate
            
        Returns:
            Tuple of (time_array, position_array)
        """
        from scipy.integrate import solve_ivp
        
        def system(t, state):
            """Convert second-order ODE to first-order system."""
            x, v = state
            m, c, k = self.params.mass, self.params.damping_coefficient, self.params.spring_constant
            
            # dx/dt = v
            # dv/dt = (-c*v - k*x) / m
            return [v, (-c * v - k * x) / m]
        
        # Initial conditions
        initial_state = [self.params.initial_position, self.params.initial_velocity]
        
        # Solve using scipy
        solution = solve_ivp(
            system, 
            t_span, 
            initial_state, 
            t_eval=np.linspace(t_span[0], t_span[1], num_points),
            method='RK45'
        )
        
        return solution.t, solution.y[0]
    
    def get_solution_function(self) -> Callable[[float], float]:
        """
        Get a callable function that evaluates the symbolic solution.
        
        Returns:
            Function that takes time and returns position
        """
        if self._symbolic_solution is None:
            self.solve_symbolic()
        
        # Convert symbolic solution to a callable function
        t = sp.Symbol('t')
        solution_func = sp.lambdify(t, self._symbolic_solution, modules=['numpy'])
        
        return solution_func
    
    def generate_sample_data(self, t_span: Tuple[float, float], num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate sample data for the oscillator.
        
        Args:
            t_span: Tuple of (start_time, end_time)
            num_points: Number of points to generate
            
        Returns:
            Tuple of (time_array, position_array)
        """
        return self.solve_numerical(t_span, num_points)
    
    def plot_solution(self, t_span: Tuple[float, float], num_points: int = 1000, 
                     show_analytical: bool = True, show_numerical: bool = True,
                     title: Optional[str] = None) -> None:
        """
        Plot the solution of the oscillator.
        
        Args:
            t_span: Tuple of (start_time, end_time)
            num_points: Number of points to generate
            show_analytical: Whether to show analytical solution
            show_numerical: Whether to show numerical solution
            title: Optional title for the plot
        """
        t_numerical, x_numerical = self.solve_numerical(t_span, num_points)
        
        plt.figure(figsize=(12, 8))
        
        if show_numerical:
            plt.plot(t_numerical, x_numerical, 'b-', linewidth=2, label='Numerical Solution')
        
        if show_analytical:
            try:
                solution_func = self.get_solution_function()
                x_analytical = solution_func(t_numerical)
                plt.plot(t_numerical, x_analytical, 'r--', linewidth=2, label='Analytical Solution')
            except Exception as e:
                print(f"Could not plot analytical solution: {e}")
        
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.title(title or f'Linear Oscillator with Damping\nm={self.params.mass}, c={self.params.damping_coefficient}, k={self.params.spring_constant}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def get_system_properties(self) -> dict:
        """
        Calculate and return system properties.
        
        Returns:
            Dictionary containing system properties
        """
        m, c, k = self.params.mass, self.params.damping_coefficient, self.params.spring_constant
        
        # Natural frequency (undamped)
        omega_n = np.sqrt(k / m)
        
        # Damping ratio
        zeta = c / (2 * np.sqrt(m * k))
        
        # Damped frequency
        if zeta < 1:
            omega_d = omega_n * np.sqrt(1 - zeta**2)
        else:
            omega_d = 0
        
        # Time constant for exponential decay
        if zeta > 0:
            tau = 1 / (zeta * omega_n)
        else:
            tau = float('inf')
        
        return {
            'natural_frequency': omega_n,
            'damping_ratio': zeta,
            'damped_frequency': omega_d,
            'time_constant': tau,
            'system_type': self._classify_system(zeta)
        }
    
    def _classify_system(self, zeta: float) -> str:
        """Classify the system based on damping ratio."""
        if zeta < 1:
            return "Underdamped"
        elif zeta == 1:
            return "Critically Damped"
        else:
            return "Overdamped"


def create_sample_oscillators() -> dict:
    """
    Create sample oscillator configurations for demonstration.
    
    Returns:
        Dictionary of sample oscillators
    """
    return {
        'underdamped': OscillatorParameters(
            mass=1.0,
            damping_coefficient=0.5,
            spring_constant=10.0,
            initial_position=1.0,
            initial_velocity=0.0
        ),
        'critically_damped': OscillatorParameters(
            mass=1.0,
            damping_coefficient=2.0 * np.sqrt(10.0),
            spring_constant=10.0,
            initial_position=1.0,
            initial_velocity=0.0
        ),
        'overdamped': OscillatorParameters(
            mass=1.0,
            damping_coefficient=8.0,
            spring_constant=10.0,
            initial_position=1.0,
            initial_velocity=0.0
        )
    }


if __name__ == "__main__":
    # Example usage
    print("Linear Oscillator with Damping - Example")
    print("=" * 50)
    
    # Create sample oscillators
    samples = create_sample_oscillators()
    
    for name, params in samples.items():
        print(f"\n{name.upper()} OSCILLATOR:")
        print(f"Parameters: m={params.mass}, c={params.damping_coefficient}, k={params.spring_constant}")
        
        # Create solver
        solver = LinearOscillatorSolver(params)
        
        # Get system properties
        properties = solver.get_system_properties()
        print(f"System Type: {properties['system_type']}")
        print(f"Natural Frequency: {properties['natural_frequency']:.3f} rad/s")
        print(f"Damping Ratio: {properties['damping_ratio']:.3f}")
        print(f"Damped Frequency: {properties['damped_frequency']:.3f} rad/s")
        
        # Generate and plot data
        t_span = (0, 10)
        solver.plot_solution(t_span, title=f"{name.title()} Oscillator")

