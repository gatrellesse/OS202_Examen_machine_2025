import matplotlib.pyplot as plt
import numpy as np

t_seq = 0.63
t_parallel = np.array([0.63, 0.28, 0.23, 0.23])
speed_up = t_seq/t_parallel
n_threads = np.array([1,2,3,4])
# Example vectors

# Plot the graph
plt.plot(n_threads, speed_up, marker='o', linestyle='-', color='b', label="Data")

# Labels and title
plt.xlabel("Number of threads")
plt.ylabel("Speed up")
plt.title("Plot of Two Vectors")
# Show legend
plt.legend()
plt.savefig("plot3.png", dpi=300)  # High-quality PNG image
# Display the graph
plt.show()
