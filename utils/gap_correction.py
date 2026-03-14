import numpy as np

def correct_prediction_gap(actual_start, pred):
    gap = actual_start - pred[0]
    return pred + gap


def correct_future_gap(last_real_price, pred):
    gap = last_real_price - pred[0]
    return pred + gap