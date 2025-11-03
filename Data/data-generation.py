"""
Synthetic Dataset Generation using Okumura-Hata Propagation Model

This module simulates a mobile user moving through a grid covered by multiple
base stations and computes RSSI values using the Okumura-Hata path loss model.
Handover events are recorded as the user moves between coverage areas.

Author: Ashutosh Borthakur (Group 17)
Date: November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class OkumuraHataSimulator:
    """
    Simulator for mobile network signal propagation using Okumura-Hata model.
    
    Attributes:
        freq_mhz (float): Carrier frequency in MHz
        hb (float): Base station antenna height in meters
        hm (float): Mobile antenna height in meters
        area_size (float): Coverage area size in km
        num_bs (int): Number of base stations
        grid_size (int): Grid resolution for user positions
        tx_power_dbm (float): Transmit power in dBm
    """
    
    def __init__(self, freq_mhz=900, hb=30, hm=1.5, area_size=2, 
                 num_bs=4, grid_size=14, tx_power_dbm=43):
        """Initialize simulation parameters."""
        self.freq_mhz = freq_mhz
        self.hb = hb
        self.hm = hm
        self.area_size = area_size
        self.num_bs = num_bs
        self.grid_size = grid_size
        self.tx_power_dbm = tx_power_dbm
        self.bs_positions = None
        self.user_positions = None
        
    def deploy_base_stations(self, random_seed=42):
        """
        Randomly deploy base stations within coverage area.
        
        Args:
            random_seed (int): Random seed for reproducibility
            
        Returns:
            numpy.ndarray: Base station positions (num_bs x 2)
        """
        np.random.seed(random_seed)
        self.bs_positions = np.random.rand(self.num_bs, 2) * self.area_size
        print(f"Deployed {self.num_bs} base stations")
        print(f"BS Positions (km):\n{self.bs_positions}")
        return self.bs_positions
    
    def generate_user_grid(self):
        """
        Generate user positions on a regular grid.
        
        Returns:
            numpy.ndarray: User positions (grid_size^2 x 2)
        """
        x = np.linspace(0, self.area_size, self.grid_size)
        xv, yv = np.meshgrid(x, x)
        self.user_positions = np.column_stack([xv.flatten(), yv.flatten()])
        print(f"Generated {len(self.user_positions)} user positions")
        return self.user_positions
    
    def calculate_path_loss(self, distance_km):
        """
        Calculate path loss using Okumura-Hata model (urban environment).
        
        Formula:
        L_path = 69.55 + 26.16*log10(f) - 13.82*log10(h_b) - a(h_m) 
                 + [44.9 - 6.55*log10(h_b)]*log10(d)
        
        Args:
            distance_km (float): Distance between transmitter and receiver in km
            
        Returns:
            float: Path loss in dB
        """
        # Mobile antenna height correction factor (medium/small city)
        a_hm = (1.1 * np.log10(self.freq_mhz) - 0.7) * self.hm - \
               (1.56 * np.log10(self.freq_mhz) - 0.8)
        
        # Path loss calculation
        path_loss = (69.55 + 
                    26.16 * np.log10(self.freq_mhz) - 
                    13.82 * np.log10(self.hb) - 
                    a_hm + 
                    (44.9 - 6.55 * np.log10(self.hb)) * np.log10(distance_km))
        
        return path_loss
    
    def calculate_rssi(self, distance_km, min_distance=0.01):
        """
        Calculate RSSI at given distance from base station.
        
        Args:
            distance_km (float): Distance from base station
            min_distance (float): Minimum distance to avoid singularity
            
        Returns:
            float: RSSI in dBm
        """
        # Prevent singularity at very small distances
        distance_km = max(distance_km, min_distance)
        
        # Calculate path loss and RSSI
        path_loss = self.calculate_path_loss(distance_km)
        rssi = self.tx_power_dbm - path_loss
        
        return rssi
    
    def simulate_handovers(self):
        """
        Simulate user movement and compute handover events.
        
        Returns:
            pandas.DataFrame: Dataset with positions, RSSI values, and handover labels
        """
        if self.bs_positions is None:
            self.deploy_base_stations()
        if self.user_positions is None:
            self.generate_user_grid()
        
        num_points = len(self.user_positions)
        results = []
        
        # Calculate RSSI from all base stations for each user position
        for i, pos in enumerate(self.user_positions):
            rssi_values = []
            
            for bs_pos in self.bs_positions:
                distance = np.linalg.norm(pos - bs_pos)
                rssi = self.calculate_rssi(distance)
                rssi_values.append(rssi)
            
            # Connect to base station with strongest signal
            connected_bs = np.argmax(rssi_values) + 1  # 1-indexed
            
            # Determine handover event (change in connected BS)
            if i == 0:
                handover = 0
                prev_bs = connected_bs
            else:
                handover = 1 if connected_bs != prev_bs else 0
                prev_bs = connected_bs
            
            # Store results
            results.append({
                'X_km': pos[0],
                'Y_km': pos[1],
                'RSSI_BS1': rssi_values[0],
                'RSSI_BS2': rssi_values[1],
                'RSSI_BS3': rssi_values[2],
                'RSSI_BS4': rssi_values[3],
                'ConnectedBS': connected_bs,
                'Handover': handover
            })
        
        df = pd.DataFrame(results)
        
        # Print statistics
        print(f"\nDataset Statistics:")
        print(f"Total samples: {len(df)}")
        print(f"Handover events: {df['Handover'].sum()} ({df['Handover'].mean()*100:.1f}%)")
        print(f"No handover: {(df['Handover']==0).sum()} ({(df['Handover']==0).mean()*100:.1f}%)")
        
        return df
    
    def visualize_coverage(self, df, save_path='results/coverage_map.png'):
        """
        Visualize base station positions and coverage area.
        
        Args:
            df (pandas.DataFrame): Dataset with simulation results
            save_path (str): Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Coverage map with connected BS
        scatter = ax1.scatter(df['X_km'], df['Y_km'], 
                            c=df['ConnectedBS'], cmap='viridis',
                            alpha=0.6, s=50)
        ax1.scatter(self.bs_positions[:, 0], self.bs_positions[:, 1],
                   c='red', marker='^', s=200, edgecolors='black',
                   label='Base Stations', zorder=5)
        ax1.set_xlabel('X (km)')
        ax1.set_ylabel('Y (km)')
        ax1.set_title('Coverage Map - Connected Base Station')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Connected BS ID')
        
        # Plot 2: Handover events
        handover_points = df[df['Handover'] == 1]
        no_handover = df[df['Handover'] == 0]
        
        ax2.scatter(no_handover['X_km'], no_handover['Y_km'],
                   c='lightblue', alpha=0.5, s=30, label='No Handover')
        ax2.scatter(handover_points['X_km'], handover_points['Y_km'],
                   c='red', alpha=0.8, s=80, marker='*', label='Handover')
        ax2.scatter(self.bs_positions[:, 0], self.bs_positions[:, 1],
                   c='black', marker='^', s=200, edgecolors='yellow',
                   label='Base Stations', zorder=5)
        ax2.set_xlabel('X (km)')
        ax2.set_ylabel('Y (km)')
        ax2.set_title('Handover Event Locations')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Coverage map saved to {save_path}")
        plt.show()


def main():
    """Main execution function."""
    print("="*70)
    print("OKUMURA-HATA HANDOVER SIMULATION")
    print("="*70)
    
    # Initialize simulator
    simulator = OkumuraHataSimulator(
        freq_mhz=900,
        hb=30,
        hm=1.5,
        area_size=2,
        num_bs=4,
        grid_size=14,
        tx_power_dbm=43
    )
    
    # Run simulation
    print("\n1. Deploying base stations...")
    simulator.deploy_base_stations(random_seed=42)
    
    print("\n2. Generating user grid...")
    simulator.generate_user_grid()
    
    print("\n3. Simulating handovers...")
    df = simulator.simulate_handovers()
    
    # Save dataset
    output_path = 'data/raw/handover_dataset.csv'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n4. Dataset saved to {output_path}")
    
    # Display sample data
    print("\nSample Data (first 5 rows):")
    print(df.head())
    
    # Visualize coverage
    print("\n5. Generating visualizations...")
    simulator.visualize_coverage(df)
    
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
