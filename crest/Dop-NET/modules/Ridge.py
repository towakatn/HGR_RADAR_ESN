#!/usr/bin/env python3
"""
Ridge: Ridge Classifier リードアウト
alpha=1.0
"""

from sklearn.linear_model import RidgeClassifier
from .config import RIDGE_CONFIG


def create_classifier(random_state):
    """Ridge分類器を生成（random_stateは使用しない）"""
    return RidgeClassifier(alpha=RIDGE_CONFIG['alpha'])


def get_name():
    return RIDGE_CONFIG['name']
