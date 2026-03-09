# Copyright (c) ModelScope Contributors. All rights reserved.
from .acc import AccMetrics
from .embedding import InfonceMetrics, PairedMetrics
from .nlg import NlgMetrics
from .reranker import RerankerMetrics
from .cua import CuaMetrics

# Add your own metric calculation method here, use `--eval_metric xxx` to enable it
# The metric here will only be called during validation

eval_metrics_map = {
    'acc': AccMetrics,
    'nlg': NlgMetrics,
    # CUA (Computer Use Agent)
    'cua': CuaMetrics,
    # embedding
    'infonce': InfonceMetrics,
    'paired': PairedMetrics,
    # reranker
    'reranker': RerankerMetrics,
}
