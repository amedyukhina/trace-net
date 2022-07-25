import numpy as np
import pylab as plt

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
CLASSES = [
    'N/A', 'cell'
]


def plot_results(img, boxes, prob=None, classes=None):
    if classes is None:
        classes = CLASSES
    # normalize boxes
    img_w, img_h = img.shape[-2:]
    boxes = boxes.reshape(boxes.shape[0], -1, 2)
    b = boxes.numpy().copy()
    b[:, :, 1] = boxes[:, :, 0] * img_h
    b[:, :, 0] = boxes[:, :, 1] * img_w

    # normalize the image
    img = img.numpy()
    img = img - np.min(img)
    img = img.transpose(1, 2, 0)

    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    ax = plt.gca()
    colors = COLORS * 100
    for i in range(len(boxes)):

        if len(boxes[i]) != 2:
            ax.add_patch(plt.Polygon(boxes[i], fill=False, color=colors[i], linewidth=3, closed=False))
        else:
            xmin, ymin, xmax, ymax = np.array(boxes[i]).ravel()
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=colors[i], linewidth=3))
            if prob is not None:
                cl = prob[i].argmax()
                p = prob[i][cl]
            else:
                cl = -1
                p = 1
            text = f'{classes[cl]}: {p:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()
