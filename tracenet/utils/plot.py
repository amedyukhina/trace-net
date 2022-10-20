import numpy as np
import pylab as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from skimage import io
from sklearn.decomposition import PCA

from tracenet.utils.points import denormalize_points

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
CLASSES = [
    'cell'
]


def plot_traces(img, traces, return_image=False, size=6):
    img = normalize(img.numpy())
    #
    # traces = bounding_line_to_points(traces)
    # traces = denormalize_points(traces, img.shape[-2:]).reshape(-1, 4)
    traces = denormalize_points(traces.reshape(-1, 2), img.shape[-2:]).reshape(-1, 4)

    fig = plt.figure(figsize=(size, size))
    plt.imshow(img)
    ax = plt.gca()
    colors = COLORS * 100
    for i in range(len(traces)):
        trace = np.fliplr(traces[i].reshape(-1, 2))
        ax.add_patch(plt.Polygon(trace, fill=False, color=colors[i], linewidth=3, closed=False))
    plt.axis('off')
    plt.tight_layout()

    if return_image:
        return __get_canvas(fig)
    plt.show()


def plot_points(img, points, return_image=False, size=6):
    img = normalize(img.numpy())
    points = points.numpy() * np.array(img.shape)

    fig = plt.figure(figsize=(size, size))
    plt.imshow(img)
    ax = plt.gca()
    colors = COLORS * 100
    for i in range(len(points)):
        point = np.fliplr(points[i].reshape(-1, 2)).ravel()
        ax.add_patch(plt.Circle(point, radius=1, fill=False, color=colors[i]))
    plt.axis('off')
    plt.tight_layout()

    if return_image:
        return __get_canvas(fig)
    plt.show()


def plot_keypoints(img, points, labels, return_image=False, size=6):
    # normalize the image
    img = normalize(img.numpy())

    fig = plt.figure(figsize=(size, size))
    plt.imshow(img)
    ax = plt.gca()
    colors = COLORS * 100
    for i, lb in enumerate(np.unique(labels)):
        ind = np.where(labels == lb)
        ax.add_patch(plt.Polygon(np.fliplr(points[ind]), fill=False, color=colors[i], linewidth=3, closed=False))
    plt.axis('off')
    plt.tight_layout()

    if return_image:
        return __get_canvas(fig)
    plt.show()


def show_imgs(imgs, s=4, norm=True):
    fig, ax = plt.subplots(1, len(imgs), figsize=(len(imgs) * s, s))
    for i, img in enumerate(imgs):
        plt.sca(ax[i])
        im = img.numpy()
        if norm:
            im = normalize(im)
        io.imshow(im)


def pca_project(embeddings):
    """
    from https://github.com/kreshuklab/spoco

    Project embeddings into 3-dim RGB space for visualization purposes
    Args:
        embeddings: ExSpatial embedding tensor
    Returns:
        RGB image
    """
    assert embeddings.ndim == 3
    # reshape (C, H, W) -> (C, H * W) and transpose
    flattened_embeddings = embeddings.reshape(embeddings.shape[0], -1).transpose()
    # init PCA with 3 principal components: one for each RGB channel
    pca = PCA(n_components=3)
    # fit the model with embeddings and apply the dimensionality reduction
    flattened_embeddings = pca.fit_transform(flattened_embeddings)
    # reshape back to original
    shape = list(embeddings.shape)
    shape[0] = 3
    img = flattened_embeddings.transpose().reshape(shape)
    # normalize to [0, 255]
    img = 255 * (img - np.min(img)) / np.ptp(img)
    return np.moveaxis(img.astype('uint8'), 0, -1)


def normalize(img):
    img = img - img.min()
    img = img / img.max()
    return img


def __get_canvas(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    return np.frombuffer(canvas.tostring_rgb(),
                         dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
