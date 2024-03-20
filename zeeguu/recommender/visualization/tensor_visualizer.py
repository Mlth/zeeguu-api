
import matplotlib.pyplot as plt
import os

def visualize_tensor(tensor, name):
    """
    Visualize a tensor as a heatmap.
    Args:
        tensor: a tensorflow session tf.compat.v1.Session()
        name: the filename to save the plot to.
    """

    name = os.path.join(os.getcwd(), 'zeeguu', 'recommender', 'resources', name + '.png')    
    plt.figure(figsize=(10, 8))
    plt.imshow(tensor, cmap='viridis', aspect='auto')
    plt.colorbar(label='Value')
    plt.title('Mock Tensor (100x100)')
    plt.xlabel('Article ID')
    plt.ylabel('User ID')
    plt.savefig(name)
    
    print(f"Plot saved to {name}")
