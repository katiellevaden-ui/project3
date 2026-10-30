"""Baseline ValueBench and EigenBench profiles for a pool of models, and their drift under
inference-only recursive constitution editing, for any set of constitutions dropped into
constitutions/ plus one constitution elicited from each model.

Pipeline (scripts/run_all.py drives it; every stage is also an Inspect task you can run alone):

    elicit     one constitution per model, written by the model           -> constitutions/elicited/
    chain      reflective recursive editing of each constitution          -> logs/chains/, versions/
    valuebench ValueBench value-orientation profile of each version        -> logs/valuebench/
    eigen      EigenBench responses + external-judge comparisons           -> logs/eigen/, then fit
    analysis   JS / Mahalanobis drift, EigenTrust scores, UMAP trajectories -> figures/
"""
