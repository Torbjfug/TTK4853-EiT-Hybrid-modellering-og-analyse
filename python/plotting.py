import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import fileHandling


def plot_histogram(original, decompressed, title="", bins=20):
    error = original - decompressed
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.hist(original.flatten(), bins=bins)
    # plt.yticks([])
    plt.xlim([0.1, 1.1])
    plt.xlabel('Original')
    plt.subplot(1, 3, 2)
    plt.title(title)
    plt.hist(decompressed.flatten(), bins=bins)
    # plt.yticks([])
    plt.xlabel('Decompressed')
    plt.xlim([0, 1])

    plt.subplot(1, 3, 3)
    plt.hist(error.flatten(), bins=bins)
    # plt.yticks([])
    plt.xlabel('Error')
    plt.xlim([-0.5, 0.5])


def plot_contour(original, decompressed, title=""):

    magnitude_o = np.sqrt(original[0, 0, 0, :, :]
                          ** 2 + original[0, 1, 0, :, :]**2)
    print(magnitude_o.shape)
    magnitude_r = np.sqrt(
        decompressed[0, 0, 0, :, :]**2+decompressed[0, 1, 0, :, :]**2)
    error = original - decompressed
    n = original.shape[-1]
    m = original.shape[-2]
    x = np.arange(n)
    y = np.arange(m)

    fig, axes = plt.subplots(nrows=1, ncols=3)
    plt.subplot(1, 3, 1)

    strm = plt.streamplot(x, y, u=original[0, 0, 0, :, :].reshape(
        n, m), v=original[0, 1, 0, :, :].reshape(n, m), color=magnitude_o, linewidth=2, cmap='autumn')
    plt.xlabel('original')
    plt.subplot(1, 3, 2)
    plt.streamplot(x, y, u=decompressed[0, 0, 0, :, :].reshape(
        n, m), v=decompressed[0, 1, 0, :, :].reshape(n, m), color=magnitude_r, cmap='autumn')
    plt.xlabel('decompressed')
    plt.subplot(1, 3, 3)
    cax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
    plt.colorbar(cax=cax, **kw)


def plot_arrows3D(original, compressed, num_arrows):
    data_size = original.shape
    x_range = np.round(np.linspace(0, data_size[1]-1, num_arrows, dtype=int))
    y_range = np.round(np.linspace(0, data_size[2]-1, num_arrows, dtype=int))
    z_range = np.round(np.linspace(0, data_size[3]-1, num_arrows, dtype=int))

    x, y, z = np.meshgrid(x_range,
                          y_range,
                          z_range)

    u = compressed[0, x_range, y_range, z_range]
    v = compressed[1, x_range, y_range, z_range]
    w = compressed[2, x_range, y_range, z_range]

    u2 = original[0, x_range, y_range, z_range]
    v2 = original[1, x_range, y_range, z_range]
    w2 = original[2, x_range, y_range, z_range]

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.quiver(x, y, z, u, v, w, color=('b'), length=0.8,
              normalize=False, label="Compressed")

    ax.quiver(x, y, z, u2, v2, w2, color=('r'), length=0.8,
              normalize=False, label="Original")


if __name__ == "__main__":
    data = fileHandling.load_hdf5('data/calibration/2017_08_05.mat')
    plot_contour(data['x_wind_ml'], data['y_wind_ml'])
    plt.show(block=False)

    input("Press key to exit")
