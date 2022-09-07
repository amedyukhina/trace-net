import numpy as np
import pylab as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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
