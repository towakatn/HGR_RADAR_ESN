# Soli: リザバー構成と融合段階の比較

この追加実験では、総リザバーノード数、線形readoutの係数数、正則化、データ分割、乱数seed、ハイパーパラメータ候補をそろえて、リザバーのsingle / parallelと特徴量の融合段階を比較します。実装は共通モジュール `modules/fusion.py` にあり、Soliからデータと実験設定を渡します。

## 比較する条件

Soliの既定の入力は、4チャンネルのDTMとRTMの計8マップです。時刻を対応させ、各サンプルの系列をリザバーに入力し、最終状態をreadoutへ渡します。クラス数を `C`、総ノード数を `N` とします。

| 条件 | リザバーへの入力 | リザバー構成 | readout / 融合 |
| --- | --- | --- | --- |
| Single-map baseline | 1マップだけ | マップごとに単一リザバー、各 `N` ノード | 融合せず、8マップの認識率を個別に報告 |
| Single–Early | 全8マップを特徴軸で結合 | 単一リザバー、`N` ノード | 共通の線形Ridge readout |
| Parallel–Early | 各branchに全8マップの結合入力 | 独立な8リザバー、各 `N/8` ノード | 状態を結合し、共通の線形Ridge readout |
| Parallel–Intermediate | 各branchに対応する1マップ | 独立な8リザバー、各 `N/8` ノード | 状態を結合し、共通の線形Ridge readout |
| Parallel–Late | 各branchに対応する1マップ | Intermediateと同じリザバー構成 | branch別Ridgeのクラススコアをsoftmaxで変換し、確率ベクトルを統合 |

baselineの `N` は各マップの個別実験に対する予算です。8個のbaselineを同時に融合するモデルの予算ではありません。Parallelでは総ノード数がマップ数で割り切れる必要があります。

比較の読み方は次のとおりです。

- Single-map / Single–Early: 複数マップの利用による差。ただし、特定のbaselineに対する入力情報量も変わります。
- Single–Early / Parallel–Early: 同じ結合入力で、単一リザバーと独立した並列リザバーを比較します。
- Parallel–Early / Parallel–Intermediate: 並列数と総ノード数を固定し、各branchへ全マップを渡す場合と対応マップだけを渡す場合を比較します。
- Parallel–Intermediate / Parallel–Late: 同じマップ別リザバーから、共通readoutとbranch別readout・確率融合を比較します。

## 公平性のための共通設定

既定値は総ノード数 `N=400`、parallelの各branchが50ノード、readout正則化 `λ=0.001`、リザバーseed `42,43,44` です。同じ外部分割を全条件・全seedで使い、分割seedは42に固定します。

readoutは既存の線形Ridge回帰（`FeatESNReadout` の RR_L、`Ψ(r)=r`）だけを使います。学習処理を `modules.readouts.fit_ridge_readout` に共通化し、追加した5構成からも同じ実装を呼び出します。one-hot教師ラベルを `Y`、リザバー最終状態を `X` として、既存コードと同じ目的関数を使います。

\[
\lVert Y-XW\rVert_F^2+\lambda\lVert W\rVert_F^2
\]

重みは既存コードの `(XᵀX + λI)⁻¹ XᵀY` で求め、逆行列計算が失敗した場合は疑似逆行列を使います。学習件数による `λ` のスケール変更は行いません。Parallel–Lateでは各branchのreadoutにも同じ実装と `λ` を使います。同じ外側foldでは全構成の学習サンプル数も共通です。fold間で学習件数が変わると、損失和に対する正則化の相対的な強さも変わる点は既存実装の仕様です。線形readoutの学習係数は、どの条件でも合計 `N×C` 個です。全比較構成のリザバーバイアスは `bias_scaling=0.0` で無効にし、readoutにも切片を追加しません。Soliの既存single/multi構成もリザバーバイアスを無効にしています。バイアスありで保存された過去の結果は、この設定の評価結果としては使用しません。

各マップの入力特徴は、当該foldの学習サンプルの時間フレームだけで平均・標準偏差を求めて標準化します。testの統計量は使いません。入力重みの標準偏差は `input_scaling / sqrt(入力次元数)` とし、結合入力の次元増加が入力振幅を大きく変えないようにします。

乱数はモデル内のローカル生成器で管理し、入力重みと再帰重みの乱数系列を分けます。このため入力次元が変わっても、同じノード数・seed・branch番号の再帰重みは共通になります。Parallel–IntermediateとParallel–Lateは同じマップ別リザバー状態を利用します。

ノード数とreadout係数数が同じでも、singleとparallelの再帰結合数、入力結合数、計算時間は同じではありません。独立branchに分けるとbranch間の再帰結合がなくなります。この構造差を含む比較として、係数数・結合数と処理時間も記録します。

## Late fusion

branch `m`、クラス `c` の出力を次のように定義します。

\[
p_{m,c}=\operatorname{softmax}(W_m^{out}x_m/T)_c,\qquad T=1
\]

softmaxで正規化されたクラススコアであり、校正済みの事後確率とは仮定しません。温度は全条件で1に固定します。[Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html) は、softmax出力の校正と、validationデータによるtemperature scalingを扱っています。温度を最適化する実験を加える場合は、外側testを使わず、追加の探索予算もそろえる必要があります。

branch数を `M=8` として、次の4方式を同じ学習済みbranchから評価します。

| 方式 | 統合 | 解釈 |
| --- | --- | --- |
| `mean` | `q_c = sum_m p_{m,c} / M` | 等重みの算術平均。5構成の主比較に使うLate条件 |
| `product` | `q_c ∝ product_m p_{m,c}` | branchの合意を強調。1branchの小さい確率に影響されやすい |
| `geometric` | `q_c ∝ exp(sum_m log(p_{m,c}) / M)` | 幾何平均。productと同じクラスを選ぶが、正規化後の確信度は異なる |
| `max` | `q_c ∝ max_m p_{m,c}` | 各クラスについて最も強く支持するbranchを採用 |

`∝` の方式は最後にクラス方向の総和で正規化します。productとgeometricはlog-spaceで計算します。均等重みのproduct / geometricは予測クラスが同じになるため、accuracyの一致は想定される動作です。NLLとBrier scoreは確信度の違いを反映します。

sum、product、maxなどは [Kittler et al. (1998), On Combining Classifiers](https://cmp.felk.cvut.cz/~matas/papers/kittler-pami98.pdf) の標準的な融合則です。平均と積の違いは [Tax et al. (2000)](https://www.sciencedirect.com/science/article/abs/pii/S0031320399001387) でも検討されています。これらの研究で得られた優劣をSoliへ直接仮定せず、方式ごとの結果を報告します。ここでのproductは、マップ間の条件付き独立性や確率校正を保証したBayesian推論ではなく、固定の融合則です。

共通モジュールの `fuse_probabilities(..., method="mean", weights=...)` と `method="geometric"` には、固定の正の重みを指定できます。重みは内部で正規化し、評価ラベルから推定しません。既定のSoli比較は等重みです。学習する重みやstackingを試す場合は、外側train内で作ったout-of-fold予測などを使う別実験として設計してください。

## 外側評価とハイパーパラメータ探索

全条件で、同じデータと以下の外側評価を使います。

| CLI名 | 分割 |
| --- | --- |
| `50_50` | 層化50:50分割を両方向で評価 |
| `10fold` | 層化10-fold交差検証 |
| `session_split` | 各被験者のsession値を半分ずつ分ける1方向の評価。既存Soliの分割方針を維持 |
| `loso` | Leave-One-Subject-Out。評価対象の被験者を学習から除外 |

`session` は既存loaderが、接頭辞を除いた `gesture_subject_recording.h5` の3番目の整数から取得した値です。これが実際の取得sessionを表すか、収録の繰り返し番号を表すかはデータの仕様で確認する必要があります。確認前は、この分割の結果を異なる取得sessionへの汎化性能と断定しないでください。

既定の `--trials 1` は固定設定の評価で、内側探索は0回です。`--trials` を2以上にすると、各外側trainの内部で共通の候補列を評価します。内側holdoutは外側trainだけから作り、subject/sessionによる分割条件も考慮します。同じ外側foldでは全構成・全リザバーseedに同じ内側分割を使います。外側testは候補選択、標準化、重み学習に使いません。

探索では、Single-mapを8マップの平均として1群にまとめ、Single–Early、Parallel–Early、Parallel–Intermediate、Parallel–Lateの`mean`と合わせた5群のスコアを平均します。この共通スコアで**1つの設定**を選び、その設定を全構成へ適用します。したがって選択された `λ` も全構成で同じです。branchごとの独立探索や、構成ごとに異なる最良設定の選択は行いません。

Lateの追加方式は同じ状態・学習済みreadoutを再利用し、追加のリザバー探索を行いません。方式を独立に報告し、外側testの最良方式を選んで代表認識率とする運用は避けます。

## 実行方法

DTM / RTMを既存の変換スクリプトで準備した後、`crest/` から実行します。

```bash
python Soli/run_fusion.py
```

既定値を明示した例です。

```bash
python Soli/run_fusion.py \
  --total-nodes 400 \
  --regularization 0.001 \
  --seeds 42,43,44 \
  --protocols 50_50,10fold,session_split,loso \
  --trials 1
```

動作確認用の小規模実行は次のとおりです。

```bash
python Soli/run_fusion.py --quick
```

`--quick` は32ノード、seed 42、固定設定、50:50両方向評価、`--max-samples 2` を既定値にします。他のCLI指定でこれらを上書きできます。全データによる科学的比較の代わりにはなりません。

`--max-samples` は全体のサンプル件数ではありません。既存loaderの規則に従って被験者番号10未満を対象とし、各ジェスチャー・被験者の収録番号が指定値未満のファイルを読みます。既定値は25です。値2では収録番号0と1が対象となり、11ジェスチャー×10被験者の全マップが存在する場合は220サンプルです。

`--candidates path.json` は共通の探索候補を辞書のJSON配列として指定します。候補数は `--trials` と一致させます。例えば `candidates.json` を次の内容にします。

```json
[
  {"regularization": 0.001},
  {"regularization": 0.01}
]
```

```bash
python Soli/run_fusion.py --trials 2 --candidates candidates.json
```

候補に指定しない設定は共通の既定設定を使います。`--output-dir` で保存先を指定できます。全オプションの現在の説明は `python Soli/run_fusion.py --help` で確認できます。

Soliの `run_all.py` は従来7方式の評価後に、この追加比較も実行します。従来方式だけを実行する場合は次を使います。

```bash
python Soli/run_all.py --legacy-only
```

追加比較だけの実験や設定変更には `run_fusion.py` を使ってください。

## 結果と既存方式との関係

既定の出力先は `Soli/results/fusion_TIMESTAMP/` です。

- `fusion_results.json`: 実験設定と評価結果。
- `fusion_summary.csv`: 構成・マップ・Late方式・評価protocolごとの集約結果。
- `fusion_records.csv`: 各fold・seedの結果。
- `fusion_contrasts.csv`: 同じ分割・seedにおける構成間の差。
- `fusion_summary.png`: 認識率の比較グラフ。

accuracy、balanced accuracy、NLL、Brier scoreと、ノード数・係数数・処理時間を確認できます。

各条件の認識率だけでなく、同じ分割・seed上の差を確認してください。`std_accuracy` とグラフの誤差線はfoldとseedを合わせた標準偏差で、信頼区間ではありません。seedごとのfold平均と、そのseed間の標準偏差は `seed_mean_accuracy` と `std_seed_mean_accuracy` に分けて記録します。50:50両方向やCVのfoldはデータを共有するため、すべてのfoldを独立な観測とみなす統計的解釈には注意が必要です。固定設定での複数seed比較はリザバー初期化に対する変動を調べる目的です。探索を有効にした場合は、seedごとに選択された共通候補が異なる可能性もあるため、その変動も含みます。

追加比較のreadoutは既存RR_Lと同じ学習処理・正則化の定義を使います。入力標準化、入力次元に応じた重みスケール、独立したリザバー乱数系列は追加比較の共通条件なので、リザバーを含む既存方式全体の認識率を数値的に再現する実験ではありません。新しい5構成間で条件をそろえ、構成と融合段階の影響を評価します。

以前の `mean_sample_squared_error` を使った結果と、現在の `existing_RR_L` の結果は区別してください。JSONにはreadout実装と目的関数を保存します。
