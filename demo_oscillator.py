"""
Demonstration script for the Linear Oscillator with Damping solver.

This script demonstrates:
1. Different damping scenarios (underdamped, critically damped, overdamped)
2. Symbolic and numerical solutions
3. System properties analysis
4. Energy analysis
5. Phase space plots
"""

import numpy as np
import matplotlib.pyplot as plt
from linear_oscillator import (
    OscillatorParameters, 
    LinearOscillatorSolver, 
    create_sample_oscillators
)


def plot_comparison_of_damping_types():
    """Plot comparison of different damping types."""
    print("Generating comparison plot of different damping types...")
    
    samples = create_sample_oscillators()
    t_span = (0, 10)
    num_points = 1000
    
    plt.figure(figsize=(15, 10))
    
    for i, (name, params) in enumerate(samples.items()):
        solver = LinearOscillatorSolver(params)
        t, x = solver.solve_numerical(t_span, num_points)
        
        plt.subplot(2, 2, i+1)
        plt.plot(t, x, 'b-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.title(f'{name.replace("_", " ").title()} Oscillator')
        plt.grid(True, alpha=0.3)
        
        # Add system properties to plot
        properties = solver.get_system_properties()
        info_text = f"ζ = {properties['damping_ratio']:.3f}\n"
        info_text += f"ωₙ = {properties['natural_frequency']:.3f} rad/s\n"
        info_text += f"ω_d = {properties['damped_frequency']:.3f} rad/s"
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('damping_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_phase_space():
    """Plot phase space diagrams for different damping types."""
    print("Generating phase space plots...")
    
    samples = create_sample_oscillators()
    t_span = (0, 10)
    num_points = 1000
    
    plt.figure(figsize=(15, 5))
    
    for i, (name, params) in enumerate(samples.items()):
        solver = LinearOscillatorSolver(params)
        t, x = solver.solve_numerical(t_span, num_points)
        
        # Calculate velocity using finite differences
        dt = t[1] - t[0]
        v = np.gradient(x, dt)
        
        plt.subplot(1, 3, i+1)
        plt.plot(x, v, 'b-', linewidth=1.5)
        plt.plot(x[0], v[0], 'ro', markersize=8, label='Start')
        plt.plot(x[-1], v[-1], 'go', markersize=8, label='End')
        plt.xlabel('Position (m)')
        plt.ylabel('Velocity (m/s)')
        plt.title(f'{name.replace("_", " ").title()} - Phase Space')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('phase_space.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_energy_analysis():
    """Plot energy analysis for underdamped oscillator."""
    print("Generating energy analysis plot...")
    
    params = OscillatorParameters(
        mass=1.0,
        damping_coefficient=0.5,
        spring_constant=10.0,
        initial_position=1.0,
        initial_velocity=0.0
    )
    
    solver = LinearOscillatorSolver(params)
    t_span = (0, 15)
    num_points = 2000
    
    t, x = solver.solve_numerical(t_span, num_points)
    
    # Calculate velocity and energies
    dt = t[1] - t[0]
    v = np.gradient(x, dt)
    
    m, k = params.mass, params.spring_constant
    kinetic_energy = 0.5 * m * v**2
    potential_energy = 0.5 * k * x**2
    total_energy = kinetic_energy + potential_energy
    
    plt.figure(figsize=(15, 10))
    
    # Position plot
    plt.subplot(2, 2, 1)
    plt.plot(t, x, 'b-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Position vs Time')
    plt.grid(True, alpha=0.3)
    
    # Velocity plot
    plt.subplot(2, 2, 2)
    plt.plot(t, v, 'r-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity vs Time')
    plt.grid(True, alpha=0.3)
    
    # Energy plot
    plt.subplot(2, 2, 3)
    plt.plot(t, kinetic_energy, 'r-', linewidth=2, label='Kinetic Energy')
    plt.plot(t, potential_energy, 'g-', linewidth=2, label='Potential Energy')
    plt.plot(t, total_energy, 'b-', linewidth=2, label='Total Energy')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    plt.title('Energy vs Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Energy decay (log scale)
    plt.subplot(2, 2, 4)
    plt.semilogy(t, total_energy, 'b-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Total Energy (J)')
    plt.title('Energy Decay (Log Scale)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('energy_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_symbolic_vs_numerical():
    """Compare symbolic and numerical solutions."""
    print("Generating symbolic vs numerical comparison...")
    
    params = OscillatorParameters(
        mass=1.0,
        damping_coefficient=0.3,
        spring_constant=10.0,
        initial_position=1.0,
        initial_velocity=0.0
    )
    
    solver = LinearOscillatorSolver(params)
    t_span = (0, 5)
    num_points = 500
    
    t, x_numerical = solver.solve_numerical(t_span, num_points)
    
    plt.figure(figsize=(12, 8))
    
    # Plot numerical solution
    plt.plot(t, x_numerical, 'b-', linewidth=2, label='Numerical Solution')
    
    # Try to plot symbolic solution
    try:
        solution_func = solver.get_solution_function()
        x_symbolic = solution_func(t)
        plt.plot(t, x_symbolic, 'r--', linewidth=2, label='Symbolic Solution')
        
        # Calculate difference
        diff = np.abs(x_numerical - x_symbolic)
        plt.subplot(2, 1, 2)
        plt.plot(t, diff, 'g-', linewidth=1)
        plt.xlabel('Time (s)')
        plt.ylabel('|Numerical - Symbolic|')
        plt.title('Difference Between Solutions')
        plt.grid(True, alpha=0.3)
        
    except Exception as e:
        print(f"Could not generate symbolic solution: {e}")
    
    plt.subplot(2, 1, 1)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Symbolic vs Numerical Solution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('symbolic_vs_numerical.png', dpi=300, bbox_inches='tight')
    plt.show()


def demonstrate_system_properties():
    """Demonstrate system properties for different configurations."""
    print("Demonstrating system properties...")
    
    # Create various oscillator configurations
    configurations = {
        'Light Damping': OscillatorParameters(1.0, 0.1, 10.0, 1.0, 0.0),
        'Medium Damping': OscillatorParameters(1.0, 1.0, 10.0, 1.0, 0.0),
        'Heavy Damping': OscillatorParameters(1.0, 5.0, 10.0, 1.0, 0.0),
        'Critical Damping': OscillatorParameters(1.0, 2.0 * np.sqrt(10.0), 10.0, 1.0, 0.0),
        'No Damping': OscillatorParameters(1.0, 0.0, 10.0, 1.0, 0.0)
    }
    
    print("\nSystem Properties Analysis:")
    print("=" * 80)
    print(f"{'Configuration':<20} {'Type':<15} {'ζ':<8} {'ωₙ':<8} {'ω_d':<8} {'τ':<8}")
    print("-" * 80)
    
    for name, params in configurations.items():
        solver = LinearOscillatorSolver(params)
        properties = solver.get_system_properties()
        
        print(f"{name:<20} {properties['system_type']:<15} "
              f"{properties['damping_ratio']:<8.3f} "
              f"{properties['natural_frequency']:<8.3f} "
              f"{properties['damped_frequency']:<8.3f} "
              f"{properties['time_constant']:<8.3f}")


def generate_sample_data_files():
    """Generate sample data files for external analysis."""
    print("Generating sample data files...")
    
    samples = create_sample_oscillators()
    t_span = (0, 10)
    num_points = 1000
    
    for name, params in samples.items():
        solver = LinearOscillatorSolver(params)
        t, x = solver.solve_numerical(t_span, num_points)
        
        # Calculate velocity
        dt = t[1] - t[0]
        v = np.gradient(x, dt)
        
        # Save to file
        filename = f'{name}_oscillator_data.csv'
        data = np.column_stack((t, x, v))
        np.savetxt(filename, data, delimiter=',', 
                  header='Time(s),Position(m),Velocity(m/s)', 
                  comments='')
        print(f"Saved {filename}")


def main():
    """Main demonstration function."""
    print("Linear Oscillator with Damping - Demonstration")
    print("=" * 50)
    
    # Demonstrate system properties
    demonstrate_system_properties()
    
    # Generate plots
    plot_comparison_of_damping_types()
    plot_phase_space()
    plot_energy_analysis()
    plot_symbolic_vs_numerical()
    
    # Generate sample data files
    generate_sample_data_files()
    
    print("\nDemonstration completed!")
    print("Generated files:")
    print("- damping_comparison.png")
    print("- phase_space.png") 
    print("- energy_analysis.png")
    print("- symbolic_vs_numerical.png")
    print("- *_oscillator_data.csv files")


if __name__ == "__main__":
    main()

