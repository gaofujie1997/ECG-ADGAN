import numpy as np


def specify_range(signals, min_val=-1, max_val=1):
    if signals is None:
        raise ValueError("No signals data.")
    if type(signals) != np.ndarray:
        signals = np.array(signals)
    select_signals = []
    for sg in signals:
        min_sg = np.min(sg)
        max_sg = np.max(sg)

        if (min_sg >= min_val and max_sg <= max_val):
            select_signals.append(sg)

    return np.array(select_signals)
