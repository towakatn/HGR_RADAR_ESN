#!/usr/bin/env python3
"""
SVM: Support Vector Machine リードアウト
RBF kernel, C=10.0, gamma='scale'
"""

from sklearn.svm import SVC
from .config import SVM_CONFIG


def create_classifier(random_state):
    """SVM分類器を生成"""
    return SVC(
        kernel=SVM_CONFIG['kernel'],
        C=SVM_CONFIG['C'],
        gamma=SVM_CONFIG['gamma'],
        random_state=random_state
    )


def get_name():
    return SVM_CONFIG['name']
