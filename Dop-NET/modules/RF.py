#!/usr/bin/env python3
"""
RF: Random Forest リードアウト
n_estimators=300, n_jobs=-1
"""

from sklearn.ensemble import RandomForestClassifier
from .config import RF_CONFIG


def create_classifier(random_state):
    """RF分類器を生成"""
    return RandomForestClassifier(
        n_estimators=RF_CONFIG['n_estimators'],
        n_jobs=RF_CONFIG['n_jobs'],
        random_state=random_state
    )


def get_name():
    return RF_CONFIG['name']
