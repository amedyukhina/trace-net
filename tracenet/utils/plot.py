import numpy as np
import pylab as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from skimage import io
from sklearn.decomposition import PCA

from tracenet.utils.points import denormalize_points, bounding_line_to_points

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
CLASSES = [
    'cell'
]


def plot_traces(img, boxes, return_image=False, size=6):
    boxes = bounding_line_to_points(boxes)
    boxes = denormalize_points(boxes, img.shape[-2:]).reshape(-1, 4)

    # normalize the image
    img = img.numpy()
    img = img - np.min(img)
    img = img.transpose(1, 2, 0)

    fig = plt.figure(figsize=(size, size))
    plt.imshow(img)
    ax = plt.gca()
    colors = COLORS * 100
    for i in range(len(boxes)):
        box = np.fliplr(boxes[i].reshape(-1, 2))
        ax.add_patch(plt.Polygon(box, fill=False, color=colors[i], linewidth=3, closed=False))
    plt.axis('off')
    plt.tight_layout()

    if return_image:
        canvas = FigureCanvas(fig)
        canvas.draw()
        plt.axis('off')
        return np.frombuffer(canvas.tostring_rgb(),
                             dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.show()


def show_imgs(imgs, s=4):
    fig, ax = plt.subplots(1, len(imgs), figsize=(len(imgs) * s, s))
    for i, img in enumerate(imgs):
        plt.sca(ax[i])
        io.imshow(img.numpy())


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
    return img.astype('uint8')
