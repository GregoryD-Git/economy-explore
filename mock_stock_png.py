# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 08:08:16 2025

@author: d23gr
"""
import matplotlib.pyplot as plt
import numpy as np

# Generate mock stock market data
np.random.seed(42)  # For reproducibility
days = np.arange(0, 45)  # Simulating 100 days
prices = 100 + np.cumsum(np.random.normal(0, 1, len(days)))  # Random walk for stock prices

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(days, prices, color="red", linewidth=15)  # Line with increased thickness

# Add an arrow at the end
plt.annotate('', xy=(days[8], prices[8]), xytext=(days[-5], prices[-5]),
             arrowprops=dict(facecolor='black', 
                             arrowstyle='<-,head_width=2,head_length=4', 
                             lw=15))

# Remove all axes and labels
plt.axis('off')

# Save the graph to a PNG file
output_file = "mock_stock_market_arrow.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Graph saved as {output_file}")

# Optional: Show the graph
plt.show()
# Optional: Show the graph
plt.show()