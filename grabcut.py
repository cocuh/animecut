import numpy as np

import maxflow
from PIL import Image
import enum

from typing import List

from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class Label(enum.IntEnum):
    Unknown = 0
    Background = 1
    Foreground = 2
    BackgroundExpected = 3
    ForegroundExpected = 4

    @classmethod
    def is_determined(cls, mat: np.ndarray):
        return np.logical_or(mat == cls.Background, mat == cls.Foreground)

    @classmethod
    def is_background(cls, mat: np.ndarray):
        return np.logical_or(mat == cls.Background, mat == cls.BackgroundExpected)

    @classmethod
    def is_foreground(cls, mat: np.ndarray):
        return np.logical_or(mat == cls.Foreground, mat == cls.ForegroundExpected)

    @classmethod
    def is_expected(cls, mat: np.ndarray):
        return np.logical_or(mat == cls.BackgroundExpected, mat == cls.ForegroundExpected)


def calc_beta(img: np.ndarray):
    w, h, c = img.shape
    diff = 0.
    diff += np.sum(np.square(img[1:, :] - img[:-1, :]))  # up
    diff += np.sum(np.square(img[:, 1:] - img[:, :-1]))  # left
    diff += np.sum(np.square(img[1:, 1:] - img[:-1, :-1]))  # upleft
    diff += np.sum(np.square(img[1:, -1:] - img[:-1, :1]))  # upleft

    if np.isclose(diff, 0.):
        beta = 0.
    else:
        beta = 1. / (2 * diff / (4 * w * h - 3 * w - 3 * h + 2))
    return beta


def calc_weights(img: np.ndarray, beta: float, gamma: float, h:float=1/256):
    gamma_skew = gamma / np.sqrt(2)

    calc = lambda x,y: gamma * np.exp(-np.sum(beta * np.square(x-y), axis=-1) + h*np.mean((x+y)/2))
    calc_skew = lambda x,y: gamma_skew * np.exp(-np.sum(beta * np.square(x-y), axis=-1)+ h*np.mean((x+y)/2))

    left_weight = calc(img[:, 1:], img[:, :-1])
    upleft_weight = calc_skew(img[1:, 1:], img[:-1, :-1])
    up_weight = calc(img[1:, :], img[:-1, :])
    upright_weight = calc_skew(img[1:, :-1], img[:-1, 1:])
    print(left_weight.mean())

    return {
        'left': left_weight, 'up': up_weight,
        'upleft': upleft_weight, 'upright': upright_weight,
    }


class Gaussian:
    def __init__(self, dim=3, mean=None, sigma=None, epsilon=1e-8):
        if mean is None:
            mean = np.zeros((dim, 1))
        if sigma is None:
            sigma = np.eye(dim)
        self.dim = dim
        self.mean = mean

        self.sigma = sigma
        self.epsilon = epsilon

    def update(self, samples: np.ndarray):
        self.mean = np.mean(samples, axis=0)
        self.sigma = np.cov(samples, rowvar=False) + self.epsilon * np.eye(self.dim)

    def prob(self, samples: np.ndarray):
        sample_num, _ = samples.shape
        assert samples.shape == (sample_num, self.dim)
        results = multivariate_normal.pdf(samples, self.mean, self.sigma)
        return results


class GaussianMixtureModel:
    components: int
    gaussians: List[GaussianMixture]
    weights: np.ndarray

    def __init__(self, components):
        self.components = components
        self.gaussians = [Gaussian() for _ in range(self.components)]
        self.weights = np.ones((self.components,)) / self.components
        self.kmeans = None

    def update_by_kmeans(self, image, mask=None):
        assert image.ndim == 3
        if mask is None:
            mask = np.full(image.shape[:-1], True)
        assert mask.ndim == 2

        samples: np.ndarray = image[mask]  # N x 3
        kmeans = KMeans(self.components)
        labels = kmeans.fit_predict(samples)
        self.kmeans = kmeans
        for c_idx in range(self.components):
            flag = (labels == c_idx)
            self.weights[c_idx] = flag.mean()
            self.gaussians[c_idx].update(samples[flag])

    def calc_log_prob(self, image):
        size = image.shape[:-1]
        samples = image.reshape((-1, 3))
        probs = np.asarray(
            [
                g.prob(samples)
                for g in self.gaussians
                ]
        ).T.dot(self.weights)
        results = np.log(probs).reshape(size)
        return results

def predict_labels(image, comps, default=Label.ForegroundExpected, top=3):
    gmm = GaussianMixture(comps)
    vec = image.reshape((-1, 3))
    gmm.fit(vec)
    idxes = sorted([
        i
        for i in range(comps)
        ],
        key=lambda i:np.linalg.det(gmm.covariances_[i])*gmm.means_[i].mean(),
        )
    fg_idxes = idxes[:top]
    bg_idxes = idxes[-top:]
    preds = gmm.predict(vec)
    is_fg = np.in1d(preds, fg_idxes).reshape(image.shape[:-1])
    is_bg = np.in1d(preds, bg_idxes).reshape(image.shape[:-1])

    labels = np.full(image.shape[:-1], int(default))
    labels[is_fg] = int(Label.ForegroundExpected)
    labels[is_bg] = int(Label.BackgroundExpected)
    return labels


def grabcut():
    gmm_components = 5
    image_fp = Image.open('chino.jpg')
    image = np.asarray(image_fp)
    image_gray = np.asarray(image_fp.convert('L'))
    image_size = image_gray.shape

    labels = predict_labels(image, 15, 3)

    background_gmm = GaussianMixtureModel(gmm_components)
    foreground_gmm = GaussianMixtureModel(gmm_components)

    beta = calc_beta(image) / 2
    gamma = 75
    default_weight = gamma * 9
    weights = calc_weights(image, beta, gamma)

    n = 40

    for i in range(10):
        print(i)
        background_gmm.update_by_kmeans(image, Label.is_background(labels))
        foreground_gmm.update_by_kmeans(image, Label.is_foreground(labels))

        graph = maxflow.GraphFloat()

        node_ids = graph.add_grid_nodes(image_size)

        edge_params = {
            'left': (np.asarray([[0, 0, 0],
                                 [1, 0, 0],
                                 [0, 0, 0]]), node_ids[:, 1:]),
            'up': (np.asarray([[0, 1, 0],
                               [0, 0, 0],
                               [0, 0, 0]]), node_ids[1:, :]),
            'upleft': (np.asarray([[1, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, 0]]), node_ids[1:, 1:]),
            'upright': (np.asarray([[0, 0, 1],
                                    [0, 0, 0],
                                    [0, 0, 0]]), node_ids[1:, :-1]),
        }

        for direction, weight in weights.items():
            struct, ids = edge_params[direction]
            assert (weight >= 0).all()
            graph.add_grid_edges(ids, weight / 2, structure=struct, symmetric=True)

        source_weights = (
            -background_gmm.calc_log_prob(image) * Label.is_expected(labels) +
            default_weight * (labels == Label.Foreground)
        )  # non-negative matrix: H x W
        sink_weights = (
            -foreground_gmm.calc_log_prob(image) * Label.is_expected(labels) +
            default_weight * (labels == Label.Background)
        )  # non-negative matrix: H x W
        assert (sink_weights >= 0).all()
        assert (source_weights >= 0).all()
        print('back', (-background_gmm.calc_log_prob(image)).mean())
        print('fore', (-foreground_gmm.calc_log_prob(image)).mean())
        graph.add_grid_tedges(node_ids, source_weights, sink_weights)

        graph.maxflow()
        segments = graph.get_grid_segments(node_ids)

        labels[segments*Label.is_expected(labels)]= int(Label.BackgroundExpected)
        labels[np.logical_not(segments)*Label.is_expected(labels)] = int(Label.ForegroundExpected)

        img2 = image.copy()
        img2[Label.is_background(labels)] = 0
        Image.fromarray(img2, mode='RGB').save(f'result{i:04d}-fg.png')

        img2 = image.copy()
        img2[Label.is_foreground(labels)] = 0
        Image.fromarray(img2, mode='RGB').save(f'result{i:04d}-bg.png')


def grabcut_a2():
    img = np.asarray(Image.open('a2.png'))
    graph = maxflow.GraphFloat()
    node_ids = graph.add_grid_nodes(img.shape)
    graph.add_grid_edges(node_ids, 50)
    graph.add_grid_tedges(node_ids, img, 255 - img)
    graph.maxflow()
    segments = graph.get_grid_segments(node_ids)
    img2 = np.logical_not(segments).astype(np.uint8) * 255
    Image.fromarray(img2, mode='L').save('result.png')


if __name__ == '__main__':
    grabcut()
