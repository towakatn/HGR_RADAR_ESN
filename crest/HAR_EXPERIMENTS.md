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

Late fusionは `mean`、`product`、`geometric`、`max` を等重みで個別に報告します。主比較は `mean` です。全構成でリザバーバイアスを0にし、readoutにも切片を追加しません。学習するreadout係数の合計は `400 × クラス数` です。ノード数は共通でも、リザバー内部の接続数・入力重み数は構造により異なるため、結果にも記録します。

| パラメータ | 既定値 |
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
| 入力標準化 | 当該訓練データの時間フレームだけで平均・標準偏差を計算 |
| softmax温度 | 1.0 |
| リザバーseed | 42, 43, 44 |
| 分割seed | 42 |
| ハイパーパラメータ探索 | なし。`n_trials=1` は固定設定の評価 |

リッジ回帰は既存実装と同じ `(XᵀX + λI)⁻¹ XᵀY` を使い、逆行列が計算できない場合は疑似逆行列へ切り替えます。学習件数によるλのスケール変更は行いません。

HARの既定λは、H1・D1の診断で得た訓練内検証のスコアを基に、0.001から0.1へ変更しました。8候補を比較し、5種類の構成を等重みで集計した検証スコアは、2分割方向 × 3seedの平均でλ=0.1が最高でした。Single-mapを含む全構成で同じλ=0.1を使用します。Soliの正則化設定は変更しません。

## 学習・テストの分割

既定の `50_50` は、選んだ1部屋・1距離の収録サンプルを行動クラスで層化して50:50に分割します。H1・D1の対象は600サンプル（3被験者 × 10行動 × 20収録）で、学習300・テスト300です。学習・テストを入れ替えた2方向について、リザバーseedを3つ使用し、各条件6回を評価します。全構成と全seedで同じ分割を使い、外側テストデータを標準化や学習に使いません。

この分割では同じ被験者の別収録が学習とテストの両方に含まれます。評価するのは、同じ部屋・同じ距離・既知の被験者を含む条件での別サンプル認識率です。未知の部屋・距離・人物への認識率としては解釈しません。必要なら `--protocols loso` で同じ部屋・同じ距離の被験者分離評価、`--protocols 10fold` で層化交差検証を実行できます。繰り返し番号を収録sessionとみなした分割は行いません。

## 実行方法

16-bitデータ本体を準備した後、`crest/` から実行します。Git LFSのポインタだけがある場合は、選んだ部屋の16-bitデータを取得します。取得処理ではファイルサイズとSHA-256を確認します。この取得コマンドはH1の全距離1,800サンプルを対象とし、実験runnerがD1の600サンプルに絞ります。

```bash
.venv/bin/python -m modules.har_download --room 1
```

```bash
python -m modules.har_experiment \
  --room 1 \
  --distance 1 \
  --total-nodes 400 \
  --regularization 0.1 \
  --seeds 42,43,44 \
  --protocols 50_50
```

`--base-dir` でHARデータセットのルート、`--output-dir` で結果の保存先を指定できます。runner自体はネットワーク取得を行いません。ローカルの薄いラッパー `python HAR-Dataset-Project/run_fusion.py` からも同じ実験を実行できます。ラッパーやデータがGit管理されていなくても、上記の共通モジュールから実行できます。

## 保存する結果

既定の保存先は `HAR-Dataset-Project/results/fusion_room1_distance1_TIMESTAMP/` です。

- `fusion_results.json`: 設定、入力ファイル・メタデータ・データmanifest、分割、各fold・seedの評価。
- `fusion_summary.csv`: 各構成のaccuracy、balanced accuracy、NLL、Brier scoreの平均と標準偏差。
- `fusion_records.csv`: 各fold・seedの数値、ノード数・重み数、処理時間。
- `fusion_contrasts.csv`: 同じ分割・seedでの構成間認識率差。
- `fusion_summary.png`: 認識率の比較グラフ。
- `split_manifest.csv`: ファイル名、部屋、被験者、行動、距離、繰り返し番号と各分割でのtrain/test所属。
- `dataset_summary.json`: 使用したサンプル数、クラス・部屋・被験者・距離の分布、マップ形状。

グラフの誤差線と `std_accuracy` はfoldとseedをまとめた標準偏差です。信頼区間ではありません。Soliと同じ実装・比較手順を使いますが、正則化係数、データセットの行動・収録条件が異なるので、認識率の単純な優劣だけでデータセット間を比較しないでください。

`results/download_H1_manifest.json` などの取得manifestが存在する場合は、その内容を `configuration.source_download_manifest` に保存します。このmanifestにはH1の全距離が含まれますが、実験に使った600件は `sample_filenames` と `sample_metadata` で確認できます。取得元のcommitと各ファイルの期待SHA-256・サイズを追跡できます。SHA-256の検証は取得時に行い、数値実験の開始時にデータ全体を再ハッシュする処理はありません。


## 融合が改善しなかった原因の診断

元のH1・D1の400ノード・λ=0.001・seed42/43/44の結果を固定し、`modules.har_fusion_diagnostics` で訓練／テスト認識率、リッジの実効自由度、マップの類似度、各Late枝の認識率を調べます。λは同じ8候補から外側訓練データ内の225件／75件分割で選び、全構成へ同じ候補を適用します。外側テストからλは選びません。既存RR_Lとバイアス0を維持します。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m modules.har_fusion_diagnostics
```

容量診断ではRTM_ch1のノード数だけを50・100・200・400に変えます。この診断と、既存枝から一部のマップだけを確率融合する診断は、総ノード数固定の主比較とは別です。元の認識率・NLL・Brier scoreの再現を確認してから追加評価し、元の結果と設定は変更しません。

`HAR-Dataset-Project/results/fusion_diagnostics_room1_distance1/` に診断レポート、比較図、元と訓練内でλを選んだ場合の集計、λ別曲線、容量診断、選択履歴、状態キャッシュを保存します。外側テストのλ別曲線は探索的な診断として扱います。
