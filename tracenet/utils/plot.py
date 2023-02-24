import numpy as np
import pylab as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from skimage import io

from tracenet.utils.points import denormalize_points

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
CLASSES = [
    'cell'
]


def plot_traces(img, traces, return_image=False, size=6, n_points=2, ax=None):
    '''
    Plotting of the traces return by TraceNet on top of the original image.
    Input traces and images are provided as torch tensors. Traces are normalized between 0 and 1.
    '''
    img = normalize(img.numpy())

    traces = denormalize_points(traces.reshape(-1, 2), img.shape[-2:]).reshape(-1, n_points * 2)

    if ax is None:
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


def plot_keypoints(img, points, labels, return_image=False, size=6):
    '''
    Plot keypoints (debugging).
    Points are provided in pixel coordinates.
    '''
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


def show_imgs(imgs, s=4, norm=True, titles=None):
    if titles is None:
        titles = [''] * len(imgs)
    fig, ax = plt.subplots(1, len(imgs), figsize=(len(imgs) * s, s))
    for i, img in enumerate(imgs):
        plt.sca(ax[i])
        im = img.numpy()
        if norm:
            im = normalize(im)
        io.imshow(im)
        plt.sca(ax[i])
        plt.title(titles[i])


def normalize(img):
    img = img - img.min()
    img = img / img.max()
    return img


def __get_canvas(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    return np.frombuffer(canvas.tostring_rgb(),
                         dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
