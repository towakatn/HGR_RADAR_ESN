"""共有の分類器生成関数。設定値はデータセット側から渡す。"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC


def create_classifier(classifier_type, config=None, *, random_state=None):
    """名前・説明を除いた設定で分類器を生成する。

    configに乱数シードがある場合はその値を優先し、指定がない場合は
    呼び出し側のrandom_stateを使用する。
    """
    params = dict(config or {})
    params.pop('name', None)
    params.pop('description', None)
    if random_state is not None:
        params.setdefault('random_state', random_state)
    classes = {
        'rf': RandomForestClassifier,
        'svm': SVC,
        'ridge': RidgeClassifier,
    }
    try:
        classifier_class = classes[classifier_type]
    except KeyError:
        raise ValueError(f"Unknown classifier type: {classifier_type}") from None
    return classifier_class(**params)


def classifier_factory(classifier_type, config, *, use_random_state=True):
    """評価関数が渡す乱数シードで分類器を作るファクトリを返す。

    use_random_state=Falseでは、設定内のシードも使用しない。
    """
    params = dict(config)
    params.pop('random_state', None)

    def create_fn(random_state):
        seed = random_state if use_random_state else None
        return create_classifier(classifier_type, params, random_state=seed)

    return create_fn
