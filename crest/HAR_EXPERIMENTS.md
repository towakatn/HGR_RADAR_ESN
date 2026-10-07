# HAR: 同じ部屋・同じ距離の16-bitデータによる比較

この実験は `HAR-Dataset-Project/Human activity recognition V2.0_Clipping/` の高精度データだけを使います。`Human Activity_1bit v2.0_Clipping/` は対象外です。ファイル名の `H` と `D` がそれぞれ同じ収録だけを選び、その中で学習データとテストデータを分けます。既定では `--room 1 --distance 1`、すなわち部屋H1・距離D1（1.5m）を選びます。距離IDはD1=1.5m、D2=3.5m、D3=5.5mです。実装と設定はGitで管理される `modules/har_experiment.py`、`modules/har_config.py` にあります。

元の `.npy` は `(受信チャネル4, 時間128, range32, Doppler32)` のテンソルです。高精度側の実際の保存dtypeはfloat64です。振幅 `abs(tensor)` を求め、range軸の総和をDTM、Doppler軸の総和をRTMとして作ります。`DTM_ch0, RTM_ch0, …, DTM_ch3, RTM_ch3` の8マップを共通の比較モジュールへ渡します。各マップは `(時間128, 特徴32)` で、時間フレームやチャネルを別サンプルに分割しません。共通loader `modules/har_data.py` が処理し、読み込んだファイルの形状・dtype・マップ生成方法は結果JSONの `configuration.data_manifest` に保存します。

## 比較する構成と共通条件

Soliで追加した比較と同じ `modules/fusion.py`、`modules/fusion_evaluation.py`、既存RR_Lのリッジ回帰を使います。

| 構成 | 入力と融合 | リザバー |
| --- | --- | --- |
| Single-map baseline | 1マップだけを入力。8マップの認識率を個別に報告 | 各マップの個別実験で400ノード |
| Single–Early | 全8マップを結合して入力 | 単一400ノード |
| Parallel–Early | 全8マップの結合入力を全branchへ渡す。最終状態を結合し、共通readout | 独立50ノード × 8 |
| Parallel–Intermediate | 各branchに対応マップを入力。最終状態を結合し、共通readout | 独立50ノード × 8 |
| Parallel–Late | マップ別リザバー・readoutのスコアをsoftmaxに変換し、確率を融合 | 独立50ノード × 8 |

Late fusionは `mean`、`product`、`geometric`、`max` を等重みで個別に報告します。主比較は `mean` です。既定ではSingle-mapの各マップ、Earlyの各構成、Intermediate、Lateの各融合則を同じ60回の予算で別々にベイズ最適化します。全構成でリザバーバイアスを0にし、readoutにも切片を追加しません。学習するreadout係数の合計は `400 × クラス数` です。ノード数は共通でも、リザバー内部の接続数・入力重み数は構造により異なるため、結果にも記録します。

| パラメータ | 固定値または探索開始値 |
| --- | --- |
| 総リザバーノード数 | 400 |
| readout | 既存の線形リッジ回帰 RR_L のみ |
| readout正則化 λ | 0.1 |
| スペクトル半径 | 0.95 |
| input scaling | 0.2。入力重みの標準偏差は `0.2 / sqrt(入力次元数)` |
| 接続密度 | 0.1 |
| リーク率 | 0.05 |
| バイアス | 0.0 |
| 活性化 | tanh |
| 使用するリザバー状態 | 各サンプルの最終時刻の状態 |
| 入力標準化 | 初期候補では有効。有無を探索し、有効なら当該訓練データの時間フレームだけで平均・標準偏差を計算 |
| softmax温度 | 1.0 |
| リザバーseed | 42, 43, 44 |
| 分割seed | 42 |
| ハイパーパラメータ探索 | `search_strategy='bayesian'`、各方式60回 |

リッジ回帰は既存実装と同じ `(XᵀX + λI)⁻¹ XᵀY` を使い、逆行列が計算できない場合は疑似逆行列へ切り替えます。学習件数によるλのスケール変更は行いません。

HARの既定λは、H1・D1の過去の診断で得た訓練内検証のスコアを基に、0.001から0.1へ変更しました。8候補を比較し、5種類の構成を等重みで集計した検証スコアは、2分割方向 × 3seedの平均でλ=0.1が最高でした。現在のベイズ最適化ではλを探索せず、Single-mapを含む全構成で同じλ=0.1を使用します。Soliの正則化設定は変更しません。

## 方式別のベイズ最適化

各外側fold・リザバーseedごとに、外側train内で共通のholdoutを作り、全方式に同じ内側trainとvalidationを渡します。LOSOの場合は内側でも被験者を分離します。EarlyとLateは、それぞれ自分のvalidationスコアで別々の最良設定を選びます。選択後は外側train全体で再学習し、外側testを評価します。

| 探索パラメータ | 範囲 |
| --- | --- |
| `spectral_radius` | 0.1–1.5、対数尺度 |
| `input_scaling` | 0.01–2、対数尺度 |
| `density` | 0.01–1、対数尺度 |
| `leakage_rate` | 0.001–1、対数尺度 |
| `temperature` | 0.1–10、対数尺度 |
| `standardize_inputs` | 無効／有効 |

総ノード数400、parallel各branchの50ノード、正則化λ=0.1は全方式で固定します。バイアス設定は設けず、常時0です。方式内の全branchは一組のパラメータを共有し、branchごとの追加探索はしません。温度はEarlyの予測クラスを変えませんが、同じ認識率の場合に優先するNLLには影響します。

60回のうち、最初の12回は設定の初期値1候補とLatin hypercubeによる11候補です。続く48回は [scikit-learnのGaussian process](https://scikit-learn.org/stable/modules/gaussian_process.html) のMatérnカーネルとExpected Improvementで候補を選びます。認識率が最も高く、同点ならNLLが最も小さい候補を採用します。初期候補を探索回数に含め、途中打ち切りやpruningを使わず、Lateの各融合則にも同じ回数を与えます。

60回は初期探索と適応的探索を確保する実用的な予算です。変更するときも `--trials` の一つの値を全方式へ適用します。既定の片方向・3seedでは、15方式 × 60回 × 3seedで1部屋につき2,700回の内側学習に、最終再学習が加わります。H1・H2・H3をそれぞれ評価すると計8,100回です。

得られる構成間の差は、同じ探索予算でそれぞれ調整した後の性能差です。IntermediateとLateも最良リザバーパラメータが異なり得るため、純粋なreadoutだけの効果とは解釈しません。同じパラメータで構造差を調べる場合は `--search fixed` を使います。外側testで最良のLate融合則を選んで代表認識率にすることはせず、各方式を個別に報告します。

## 学習・テストの分割

既定の `50_50` は、選んだ1部屋・1距離の収録サンプルを行動クラスで層化して50:50に分割し、片方向だけ評価します。H1・D1は600サンプルで学習300・テスト300、H2・D1は598サンプルで299・299、H3・D1は580サンプルで290・290です。収録ファイルの実際の件数を使用します。リザバーseedを3つ使用し、各条件3回を評価します。全構成と全seedで同じ分割を使い、外側テストデータを標準化や学習に使いません。`--bidirectional` で両方向の評価に切り替えられます。共通APIでは `bidirectional_50_50=False` を指定します。Soliの既定は両方向のままです。

この分割では同じ被験者の別収録が学習とテストの両方に含まれます。評価するのは、同じ部屋・同じ距離・既知の被験者を含む条件での別サンプル認識率です。未知の部屋・距離・人物への認識率としては解釈しません。必要なら `--protocols loso` で同じ部屋・同じ距離の被験者分離評価、`--protocols 10fold` で層化交差検証を実行できます。繰り返し番号を収録sessionとみなした分割は行いません。

## 実行方法

16-bitデータ本体を準備した後、`crest/` から実行します。Git LFSのポインタだけがある場合は、部屋と距離を指定して16-bitデータを取得します。取得処理ではファイルサイズとSHA-256を確認します。`--distance 1` はD1だけを取得し、距離指定を省略するとその部屋の全距離を取得します。

```bash
.venv/bin/python -m modules.har_download --room 1 --distance 1
.venv/bin/python -m modules.har_download --room 2 --distance 1
.venv/bin/python -m modules.har_download --room 3 --distance 1
```

```bash
python -m modules.har_experiment \
  --room 1 \
  --distance 1 \
  --total-nodes 400 \
  --regularization 0.1 \
  --seeds 42,43,44 \
  --protocols 50_50 \
  --one-way \
  --search bayesian \
  --trials 60
```

`--base-dir` でHARデータセットのルート、`--output-dir` で結果の保存先を指定できます。runner自体はネットワーク取得を行いません。ローカルの薄いラッパー `python HAR-Dataset-Project/run_fusion.py` からも同じ実験を実行できます。ラッパーやデータがGit管理されていなくても、上記の共通モジュールから実行できます。

### 同じ部屋で距離を比較する

部屋H1に固定し、`--distance 1`（1.5m）、`--distance 2`（3.5m）、`--distance 3`（5.5m）を別々に実行します。H1は各距離600件で、訓練300件・テスト300件です。各距離内で学習・テストし、距離ごとに全方式を独立にベイズ最適化します。総ノード数400、共通λ=0.1、分割seed=42、リザバーseed=42/43/44、各方式60回探索を維持します。

```bash
for har_distance in 1 2 3; do
  .venv/bin/python -m modules.har_download --room 1 --distance "$har_distance"
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m modules.har_experiment \
    --room 1 --distance "$har_distance" --one-way \
    --total-nodes 400 --regularization 0.1 --seeds 42,43,44 \
    --search bayesian --trials 60
done
```

H1の3距離は被験者・行動・繰り返し番号の組み合わせが一致するため、同じ分割seedで対応する収録が同じ側に入ります。距離別に最適化した後の、各距離内の別収録の認識率を比較します。λ=0.1は過去のH1・D1の検証で選ばれた既存設定を全距離へ共通に適用します。

2026-10-07の比較結果は `HAR-Dataset-Project/results/bayesian_H1_D1_D2_D3_oneway_20261007/` に保存しています。全1,800件のソースSHA256・サイズを再照合し、D1は同条件の既存2,700回を引き継ぎ、D2・D3は計5,400回を新規に実行しました。`accuracy_report.md`、`accuracy_by_distance.png`、`distance_deltas.csv`、`selected_hyperparameters.csv`、`audit.json` に精度・距離差・選択設定・検証結果をまとめています。

固定設定で評価する場合は `python -m modules.har_experiment --search fixed` を実行します。この場合は内側探索を行いません。従来の共通候補比較は `--search shared_candidates --candidates candidates.json` で実行できます。`--trials` を省略すれば候補配列の長さを回数にします。共通候補モードでは一つの候補を選んで全構成へ同じ設定を適用します。

読み込み済みのデータから共通モジュールを呼び出す例です。

```python
from modules.fusion_evaluation import run_fusion_comparison

results = run_fusion_comparison(
    maps, y, metadata,
    total_nodes=400, regularization=0.1,
    search_strategy="bayesian", n_trials=60,
    bidirectional_50_50=False,
    seeds=(42, 43, 44), protocols=("50_50",),
)
```

共通モジュールの `search_strategy` を省略した既存のPython呼び出しは、候補配列または `n_trials>1` なら共通候補モード、それ以外なら固定モードとして扱います。HAR runnerへ渡す部分設定も、`n_trials=1` を明示した旧設定は固定モードを維持します。

## 保存する結果

既定の保存先は `HAR-Dataset-Project/results/fusion_room1_distance1_TIMESTAMP/` です。

- `fusion_results.json`: 設定、入力ファイル・メタデータ・データmanifest、分割、方式別の探索履歴と選択パラメータ、各fold・seedの評価。
- `fusion_summary.csv`: 各構成のaccuracy、balanced accuracy、NLL、Brier scoreの平均と標準偏差。
- `fusion_records.csv`: 各fold・seedの数値、ノード数・重み数、処理時間。
- `fusion_contrasts.csv`: 同じ分割・seedでの構成間認識率差。
- `fusion_summary.png`: 認識率の比較グラフ。
- `split_manifest.csv`: ファイル名、部屋、被験者、行動、距離、繰り返し番号と各分割でのtrain/test所属。
- `dataset_summary.json`: 使用したサンプル数、クラス・部屋・被験者・距離の分布、マップ形状。

グラフの誤差線と `std_accuracy` はfoldとseedをまとめた標準偏差です。信頼区間ではありません。Soliと同じ実装・比較手順を使いますが、正則化係数、データセットの行動・収録条件が異なるので、認識率の単純な優劣だけでデータセット間を比較しないでください。

取得manifestは距離別の `results/download_H2_D1_manifest.json` などを優先し、なければ従来の部屋別 `download_H1_manifest.json` を使用します。その内容を `configuration.source_download_manifest` に保存します。実験に使ったファイルは `sample_filenames` と `sample_metadata` で確認できます。取得元のcommitと各ダウンロードファイルの期待SHA-256・サイズを追跡できます。SHA-256の検証は取得時に行います。

部屋全体が取得済みなら、距離指定のダウンロード処理は再取得を行わず、同じソースcommitの部屋別manifestから、その距離に該当する元ハッシュ・サイズを距離別manifestへ引き継ぎます。


## 融合が改善しなかった原因の診断

元のH1・D1の400ノード・λ=0.001・seed42/43/44の固定設定結果を使い、`modules.har_fusion_diagnostics` で訓練／テスト認識率、リッジの実効自由度、マップの類似度、各Late枝の認識率を調べます。現在のベイズ最適化による主比較とは別の診断です。λは同じ8候補から外側訓練データ内の225件／75件分割で選び、全構成へ同じ候補を適用します。外側テストからλは選びません。既存RR_Lとバイアス0を維持します。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m modules.har_fusion_diagnostics
```

容量診断ではRTM_ch1のノード数だけを50・100・200・400に変えます。この診断と、既存枝から一部のマップだけを確率融合する診断は、総ノード数固定の主比較とは別です。元の認識率・NLL・Brier scoreの再現を確認してから追加評価し、元の結果と設定は変更しません。

`HAR-Dataset-Project/results/fusion_diagnostics_room1_distance1/` に診断レポート、比較図、元と訓練内でλを選んだ場合の集計、λ別曲線、容量診断、選択履歴、状態キャッシュを保存します。外側テストのλ別曲線は探索的な診断として扱います。
