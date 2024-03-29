import numpy as np
import pytest
import torch

from tracenet.losses.accuracy_metric import Metric


@pytest.fixture(params=[2, 3, 5])
def n_points(request):
    return request.param


@pytest.fixture(params=[2, 3])
def trace_pair(request, n_points):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    n_traces = np.random.randint(5, 20, request.param)
    targets = dict({'trace': [torch.rand(n, n_points * 2).to(device) for n in n_traces],
                    'trace_class': [torch.ones(n).to(device).long() for n in n_traces]})
    outputs = dict({'pred_traces': torch.rand(len(n_traces), 100, n_points * 2).to(device),
                    'pred_logits': torch.rand(len(n_traces), 100, 2).to(device)})

    for i in range(len(n_traces)):
        len_trace = targets['trace'][i].shape[0]
        outputs['pred_traces'][i][:len_trace] = targets['trace'][i]
        outputs['pred_logits'][i][:len_trace, 0] = -100
        outputs['pred_logits'][i][:len_trace, 1] = 100

    return outputs, targets


@pytest.fixture
def trace_pair_bezier(n_points):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    n_traces = np.random.randint(5, 20, 3)
    targets = dict({'trace': [torch.rand(n, n_points * 2).to(device) for n in n_traces],
                    'trace_class': [torch.ones(n).to(device).long() for n in n_traces]})
    outputs = dict({'pred_traces': torch.rand(len(n_traces), 100, 8).to(device),
                    'pred_logits': torch.rand(len(n_traces), 100, 2).to(device)})

    return outputs, targets


def test_metric(trace_pair):
    metric = Metric(min_prob=0.99)
    outputs, targets = trace_pair
    metric(outputs, targets)
    metric(outputs, targets)
    assert metric.buffer['cardinality error'].shape[0] == 2 * len(targets['trace'])
    metric.aggregate()
    for key in ['cardinality error', 'relative cardinality error',
                'Precision', 'Recall', 'F1 Score', 'end error', 'trace distance error',
                'filament length error', 'filament end error', 'curvature error']:
        assert key in metric.mean
        assert key in metric.std
        assert metric.std[key] == 0

    for key in ['Precision', 'Recall', 'F1 Score']:
        assert metric.mean[key] == 1
    for key in ['cardinality error', 'relative cardinality error', 'end error', 'trace distance error',
                'filament length error', 'filament end error', 'curvature error']:
        assert metric.mean[key] == 0

    metric.reset()
    assert len(metric.buffer.keys()) == 0
    outputs['pred_logits'][0, 0] = 0.5
    metric(outputs, targets)
    metric.aggregate()
    assert round(metric.mean['cardinality error'], 5) == round(1. / len(targets['trace']), 5)
    assert metric.std['cardinality error'] >= 0
    for key in ['Recall', 'F1 Score']:
        assert metric.mean[key] <= 1
    assert metric.mean['Precision'] == 1
    for key in ['end error', 'trace distance error', 'filament length error',
                'filament end error', 'curvature error']:
        assert metric.mean[key] == 0


def test_metric_bezier(trace_pair_bezier):
    metric = Metric(min_prob=0.5, bezier=True)
    outputs, targets = trace_pair_bezier
    metric(outputs, targets)
    metric.aggregate()
    for key in ['cardinality error', 'relative cardinality error',
                'Precision', 'Recall', 'F1 Score', 'end error', 'trace distance error',
                'filament length error', 'filament end error', 'curvature error']:
        assert key in metric.mean
        assert key in metric.std
