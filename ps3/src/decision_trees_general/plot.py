import matplotlib.pyplot as plt
import numpy as np

def plot_xy_with_labels_and_line(x, y, z, k, b):
    """
    Plots x, y with labels z and draws a line y = kx + b.
    When z = 0, points are plotted as circles.
    When z = 1, points are plotted as crosses.
    
    Parameters:
    - x: Array-like, the x coordinates of the points.
    - y: Array-like, the y coordinates of the points.
    - z: Array-like, the labels of the points (0 or 1).
    - k: Slope of the line.
    - b: Y-intercept of the line.
    """
    # Plot points with labels
    for i in range(len(x)):
        if z[i] == 0:
            plt.scatter(x[i], y[i], marker='o', color='blue', label='Circle' if i == 0 else "")
        else:
            plt.scatter(x[i], y[i], marker='x', color='red', label='Cross' if i == 0 else "")

    # Calculate line values
    x_values = np.array([min(x), max(x)])
    y_values = k * x_values + b
    
    # Plot line
    plt.plot(x_values, y_values, label=f'Line: y = {k}x + {b}', color='green')
    
    # Labeling the axes
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Plot of X and Y with Z as Label and a Line')
    
    # Avoid repeating labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    # Show plot
    plt.show()

# Example usage
x = np.array([24, 53, 23, 25, 32, 52, 22, 43, 52, 48])
y = np.array([40, 52, 25, 77, 48, 110, 38, 44, 27, 65])
z = np.array([1,0,0,1,1,1,1,0,0,1])

plot_xy_with_labels_and_line(x, y, z, 1, 7.5)
