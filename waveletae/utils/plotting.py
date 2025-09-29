import matplotlib.pyplot as plt


def plot_reconstruction(original, reconstructed, idx=0):
    plt.plot(original[idx], label="Original")
    plt.plot(reconstructed[idx], label="Reconstructed")
    plt.legend()
    plt.show()
