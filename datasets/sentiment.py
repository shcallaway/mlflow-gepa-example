"""Sentiment classification dataset."""

from typing import List, Dict, Tuple


TRAIN_DATA = [
    ("This movie was absolutely fantastic! I loved every minute.", "positive"),
    ("Terrible experience. Would not recommend to anyone.", "negative"),
    ("Best purchase I've made all year! Highly recommend.", "positive"),
    ("Complete waste of time and money. Very disappointed.", "negative"),
    ("Amazing quality and fast delivery. Very happy!", "positive"),
    ("Poor customer service and broken product.", "negative"),
    ("Exceeded all my expectations. Will buy again!", "positive"),
    ("Worst meal I've ever had. Don't go there.", "negative"),
]

DEV_DATA = [
    ("This product is incredible! Worth every penny.", "positive"),
    ("Not good at all. Returned it immediately.", "negative"),
    ("Absolutely love it! Five stars!", "positive"),
    ("Horrible quality. Very upset with this purchase.", "negative"),
]


def get_data() -> Tuple[List[Dict], List[Dict]]:
    """
    Get sentiment classification train and dev datasets.

    Returns:
        Tuple of (train_data, dev_data) where each is a list of dicts with format:
        {"inputs": {"text": "..."}, "expectations": {"sentiment": "..."}}
    """
    train = []
    for text, sentiment in TRAIN_DATA:
        train.append({
            "inputs": {"text": text},
            "expectations": {"sentiment": sentiment}
        })

    dev = []
    for text, sentiment in DEV_DATA:
        dev.append({
            "inputs": {"text": text},
            "expectations": {"sentiment": sentiment}
        })

    return train, dev
