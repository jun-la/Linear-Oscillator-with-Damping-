# Linear Oscillator with Damping - SymPy Solver

This project implements a comprehensive solution for the Linear Oscillator with Damping differential equation using SymPy for symbolic computation, along with numerical methods and visualization tools.

## Overview

The Linear Oscillator with Damping is described by the differential equation:

```
m * d²x/dt² + c * dx/dt + k * x = 0
```

Where:
- `m` = mass (kg)
- `c` = damping coefficient (N·s/m)
- `k` = spring constant (N/m)
- `x` = displacement (m)
- `t` = time (s)

## Features

- **Symbolic Solution**: Uses SymPy to find exact analytical solutions
- **Numerical Solution**: Implements numerical integration using SciPy
- **System Analysis**: Calculates natural frequency, damping ratio, and system classification
- **Visualization**: Comprehensive plotting capabilities including phase space analysis
- **Energy Analysis**: Tracks kinetic, potential, and total energy over time
- **Unit Testing**: Complete test suite with pytest
- **Sample Data Generation**: Exports data for external analysis

## System Classification

The solver automatically classifies systems based on the damping ratio (ζ):

- **Underdamped** (ζ < 1): Oscillatory motion with exponential decay
- **Critically Damped** (ζ = 1): Fastest return to equilibrium without oscillation
- **Overdamped** (ζ > 1): Slow return to equilibrium without oscillation

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Eclips
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from linear_oscillator import OscillatorParameters, LinearOscillatorSolver

# Define oscillator parameters
params = OscillatorParameters(
    mass=1.0,                    # kg
    damping_coefficient=0.5,     # N·s/m
    spring_constant=10.0,        # N/m
    initial_position=1.0,        # m
    initial_velocity=0.0         # m/s
)

# Create solver
solver = LinearOscillatorSolver(params)

# Get system properties
properties = solver.get_system_properties()
print(f"System Type: {properties['system_type']}")
print(f"Damping Ratio: {properties['damping_ratio']:.3f}")

# Solve numerically
t, x = solver.solve_numerical((0, 10), 1000)

# Plot solution
solver.plot_solution((0, 10))
```

### Running the Demo

```bash
python demo_oscillator.py
```

This will generate:
- Comparison plots of different damping types
- Phase space diagrams
- Energy analysis plots
- Symbolic vs numerical solution comparison
- CSV data files for external analysis

### Running Tests

```bash
pytest test_linear_oscillator.py -v
```

## Project Structure

```
Eclips/
├── linear_oscillator.py      # Main solver implementation
├── test_linear_oscillator.py # Unit tests
├── demo_oscillator.py        # Demonstration script
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## Key Classes

### OscillatorParameters

Dataclass for storing oscillator parameters:
- `mass`: Mass of the oscillator
- `damping_coefficient`: Damping coefficient
- `spring_constant`: Spring constant
- `initial_position`: Initial displacement
- `initial_velocity`: Initial velocity

### LinearOscillatorSolver

Main solver class with methods:
- `solve_symbolic()`: Find analytical solution using SymPy
- `solve_numerical()`: Solve using numerical integration
- `get_system_properties()`: Calculate system characteristics
- `plot_solution()`: Generate plots
- `generate_sample_data()`: Export data

## System Properties

The solver calculates:
- **Natural Frequency** (ωₙ): `sqrt(k/m)`
- **Damping Ratio** (ζ): `c/(2*sqrt(m*k))`
- **Damped Frequency** (ω_d): `ωₙ * sqrt(1-ζ²)` for ζ < 1
- **Time Constant** (τ): `1/(ζ*ωₙ)` for ζ > 0

## Examples

### Underdamped Oscillator
```python
params = OscillatorParameters(1.0, 0.5, 10.0, 1.0, 0.0)
# Results in oscillatory motion with exponential decay
```

### Critically Damped Oscillator
```python
params = OscillatorParameters(1.0, 2*sqrt(10), 10.0, 1.0, 0.0)
# Results in fastest return to equilibrium without oscillation
```

### Overdamped Oscillator
```python
params = OscillatorParameters(1.0, 8.0, 10.0, 1.0, 0.0)
# Results in slow return to equilibrium without oscillation
```

## Mathematical Background

### Differential Equation
The governing equation is:
```
m * d²x/dt² + c * dx/dt + k * x = 0
```

### Solution Forms

**Underdamped (ζ < 1):**
```
x(t) = e^(-ζωₙt) * [A*cos(ω_d*t) + B*sin(ω_d*t)]
```

**Critically Damped (ζ = 1):**
```
x(t) = e^(-ωₙt) * [A + B*t]
```

**Overdamped (ζ > 1):**
```
x(t) = e^(-ζωₙt) * [A*cosh(ω_d*t) + B*sinh(ω_d*t)]
```

Where:
- ωₙ = natural frequency
- ω_d = damped frequency
- A, B = constants determined by initial conditions

## Dependencies

- **SymPy**: Symbolic mathematics
- **NumPy**: Numerical computing
- **Matplotlib**: Plotting and visualization
- **SciPy**: Scientific computing (integration)
- **pytest**: Unit testing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- SymPy community for the excellent symbolic mathematics library
- SciPy community for numerical integration tools
- Matplotlib community for visualization capabilities

