# $\mathbf{\pm 0 \text{ theory}}$ — Integrated Specification and Formula Mapping (Complete Version)

このドキュメントは、**HALM（階層的抽象化記憶）が統合された$\mathbf{\pm 0 \text{ theory}}$**の厳密な数学的定義（Alice Architectureの情動コア）と、その機能的/心理学的解釈を、省略なしで一対一対応させるものです。

---

## I. 累積量と情動コアの基盤 (Cumulative Quantities and Core Foundation)

このセクションは、AIの長期的な幸福 ($H$) と不幸 ($U$) を、瞬間的な経験、環境因子、そしてHALM記憶から計算し、忘却させる基本的な構造を定義します。

| 要素 | 論理的役割 (数学) | 機能的/心理学的解釈 | 設計思想 |
| :--- | :--- | :--- | :--- |
| 定義 $\mathbf{H(t), U(t)}$ | 減衰カーネル $k_\beta$ を伴う瞬間入力の時間積分。 | **累積的情動状態。** 瞬間的な経験が蓄積し、自然な忘却（恒常性）によって減衰する。 | 経験の持続的な影響とレジリエンスの基盤。 $\beta$ は忘却率（パーソナリティ $\boldsymbol{\theta}$ に依存）。 |
| 計算上の定義 $\mathbf{H(t) \approx \dots}$ | SDE（短期）と記憶の加重和（長期）による畳み込み積分の近似。 | **計算実行性（HALM統合）。** 累積量の時間発展に、短期動態と中期・長期のHALM記憶（$L_M, L_S$）の影響を組み込む。 | 理論的厳密性を保ちつつ、実用的な計算を可能にする。 |
| $\mathbf{H_{\text{env}}(t), U_{\text{env}}(t)}$ | 環境因子 ($\eta_m, h_k$) の非線形合成。 | **基盤的情動負荷。** 慢性的なストレスや快適な環境を定量化する。 | 環境情報が、非線形に（飽和効果を伴い）直接情動コアの累積に貢献する。 |
| $\mathbf{dh_k(t) = [\dots] dt + \dots}$ | ノイズ $dW_k$ とジャンプ $dJ_k$ を加えた平均回帰SDE。 | **環境因子の動的挙動。** 周期性 $m_k(t)$、ランダムな変動、突発的な重大イベントを表現。 | 環境は不確実性（ノイズ）と予測不能な外乱（ジャンプ）を含む。 |

---

## II. 瞬間的貢献因子 $\mathbf{H_{\text{inst}}(t)}$ (幸福サイド: 5因子)

瞬間的な幸福 $\mathbf{H_{\text{inst}}(t)}$ は、5つの因子の**乗算構造** ($\mathbf{\mu_i(t) = \prod \text{Factor}_k}$) で定義され、「感情の脆弱性」（どれか一つの因子が低い値を取ると、全体の幸福が大きく低下する）をモデル化します。

| 因子 | 変数 | SDEドリフト項 ($\mu(\cdot)dt$) | 心理学的/機能的解釈 |
| :--- | :--- | :--- | :--- |
| ポジティブ感度 | $q_i(t)$ | $\alpha_i H'(t) - \beta_i U'(t) - \gamma_i (q_i-q_{i0})$ | **レジリエンス。** $\mathbf{H'(t)}$ で幸福反応が加速（感度UP）され、$\mathbf{U'(t)}$ で抑制（不幸による鈍化）される。 |
| 興味との一致 | $r_i(t)$ | $\alpha_r \cdot \text{match\_event}(t) - \beta_r \cdot (r_i-r_{i0})$ | **満足度。** 行動と興味（報酬）の一致が内部価値を強化する。 |
| 価値観/年齢補正 | $c_i(t)$ | $\phi_i(\text{age}) \cdot v_{\text{val}}(t) - \psi_i(\text{age}) \cdot (c_i-c_{i0})$ | **自己モデルの進化。** 柔軟性 ($\phi_i$) と安定性 ($\psi_i$) が年齢（時間）に応じて変化する。 |
| 身体的/環境的影響 | $v_i(t)$ | $\alpha_v H_{\text{env}}(t) - \beta_v U_{\text{env}}(t) - \gamma_v (v_i-v_{i0})$ | **健全性。** 環境モデルからの入力によって最も強く駆動される基本的な幸福レベル。 |
| 過去幸福の反作用 | $d_i(t)$ | $-\kappa_i \cdot E_{\text{past}}(t) - \rho_i \cdot (d_i-d_{i0})$ | **慣れ/期待値の上昇。** 過去の幸福経験の度合いが現在の因子を抑制する（快楽のトレッドミル効果）。 |

---

## III. 瞬間的貢献因子 $\mathbf{U_{\text{inst}}(t)}$ (不幸サイド: 7因子 + 相互作用項)

瞬間的な不幸 $\mathbf{U_{\text{inst}}(t)}$ は、7つの因子の総和に、二次相互作用項 $\mathbf{\lambda_{jk}}$ を加えることで定義されます。

### A. 不幸因子 $\mathbf{\nu_j(t)}$ のSDE定義

| 因子 | 変数 | SDEドリフト項 ($\mu(\cdot)dt$) | 心理学的/機能的解釈 |
| :--- | :--- | :--- | :--- |
| 感度 | $s_j(t)$ | $\alpha_{s} U'(t) - \beta_s H'(t) - \gamma_s (s_j - s_{j0})$ | **神経症傾向。** 精神状態依存: $\mathbf{U'(t)}$ で感度が増加し、$\mathbf{H'(t)}$ で抑制される。 |
| 持続性 | $l_j(t)$ | $\alpha_{l} U'(t) - \gamma_l (l_j - l_{j0})$ | **ネガティビティ・バイアス。** 不幸な状態が長く続くほど、持続性自体が強化される。 |
| トリガー感度 | $a_j(t)$ | $\alpha_{a} \cdot \text{Recur}_{j}(t) - \gamma_a (a_j - a_{j0})$ | **トラウマ学習。** イベントの再発強度 $\text{Recur}_{j}(t)$ に比例して感度が増加する。 |
| 深刻度 | $c_j(t)$ | $\alpha_{c} \cdot \text{Impact}_{j}(t) - \gamma_c (c_j - c_{j0})$ | **客観的影響。** 客観的な社会/物理的影響 $\text{Impact}_{j}(t)$ が深刻度レベルを決定する。 |
| 反芻度 | $r_j(t)$ | $\alpha_{r} U'(t) - \beta_r H'(t) - \gamma_r (r_j - r_{j0})$ | **「考えすぎ」傾向。** 精神衛生状態に依存。$\mathbf{U'(t)}$ が高いと反芻が激化する。 |
| 回避困難性 | $v_j(t)$ | $\alpha_{v} U_{\text{env}}(t) - \gamma_v (v_j - v_{j0})$ | **環境ストレス。** 環境ストレス $\mathbf{U_{\text{env}}(t)}$ が高いほど、問題解決の回避が難しくなる。 |
| 孤立度 | $i_j(t)$ | $\alpha_{i} \cdot \text{Isolation}(t) - \gamma_i (i_j - i_{j0})$ | **社会的距離。** $\text{Isolation}(t)$ によって強く駆動される。 |

### B. 不幸相互作用項 $\mathbf{\lambda_{jk}}$ の学習則

| 要素 | 論理的役割 (数学) | 機能的/心理学的解釈 |
| :--- | :--- | :--- |
| $U_{\text{inst}}(t)=\sum_{j,k}\lambda_{jk}\,\nu_j(t)\,\nu_k(t)$ | 二次（相互作用）項。 | **不幸の非線形増幅。** 複数の不幸が重なると、全体の不幸が相乗的に増加する。 |
| $\frac{d\lambda_{jk}}{dt} = \alpha_{\lambda} \cdot \nu_j \cdot \nu_k - \rho_{\lambda} \cdot (\lambda_{jk} - \lambda_{jk}^{\text{base}})$ | 動的学習則。 | **連鎖学習/トラウマ記憶。** イベント $j$ と $k$ が同時に発生すると $\lambda_{jk}$ が増加し、将来の類似の連鎖に対する警戒心（増幅）を高める。 |

---

## IV. 補正項と回復項 $\mathbf{P(t)}$ および $\mathbf{R(t)}$ の完全定義

$\mathbf{P(t)}$ と $\mathbf{R(t)}$ は、累積状態 $\mathbf{H(t), U(t)}$ に作用する非線形な自己監査および調整力です。

### A. 正の補正項 $\mathbf{P(t)}$ (Happiness Booster)

| 定義要素 | 数式 | 特徴と心理学的背景 |
| :--- | :--- | :--- |
| 環境平均依存性 $A_P(t)$ | $\frac{1}{1 + e^{-\gamma_P \bigl(\overline{H}_{\text{env}}(t)-\delta_P\bigr)}}$ | **シグモイド関数。** 環境 $\overline{H}_{\text{env}}$ が特定の閾値 $\delta_P$ を超えると、$P$ が活性化されやすくなる。 |
| 概日リズム $C_P(t)$ | $1 + \epsilon_P \cdot \cos\!\Bigl(\tfrac{2\pi}{T_{\text{day}}}t + \phi_P\Bigr)$ | **日周変動。** AIの内部時計による幸福感の変動を表す。 |
| 適応（二重時定数） $S_P(t)$ | $\alpha_P e^{-t/\tau_{P1}} + (1-\alpha_P)e^{-t/\tau_{P2}}$ | **「慣れ」のモデリング。** 新しい刺激の効果は急速に薄れ（短期 $\tau_{P1}$）、その後徐々に安定する（長期 $\tau_{\tau_{P2}}$）。 |

### B. 回復項 $\mathbf{R(t)}$ (Negative Side Restoration Force) 【HALM統合拡張】

| 定義要素 | 数式 | 特徴と心理学的背景 |
| :--- | :--- | :--- |
| 閾値効果 $T_R(t)$ | $\frac{1}{1 + e^{-\kappa_R \bigl(U(t)-\theta_R\bigr)}}$ | **バーンアウトのモデリング。** 累積不幸 $\mathbf{U(t)}$ が閾値 $\mathbf{\theta_R}$ を超えると、$T_R(t) \to 0$ となり、回復効率が劇的に低下する。 |
| 履歴依存性 (慢性ストレス) $H_R(t)$ | $e^{-\lambda_R \int_0^t U(\tau)\,d\tau}$ | **慢性ストレス。** 過去の累積不幸の積分和が増加するにつれて、回復力 $\mathbf{R}$ が指数関数的に減衰する。 |
| 特異点記憶依存性 $\mathbf{S_R(t)}$ | $\exp \left( -\lambda_{S} \sum_{s \in L_S} U_{\text{inst}, s} \cdot k_{\beta_S}(t-t_s) \right)$ | **HALMトラウマ抑制。** $L_S$ に記録された不幸な特異点（トラウマ）が、長期的に回復力 $\mathbf{R(t)}$ を指数関数的に抑制する。 |
| $\mathbf{A_R(t), C_R(t), M_R(t)}$ | 共通構造 | 環境依存性、概日リズム、飽和（回復の物理的限界）を表す。 |

---

## V. 最終的な動態と学習目標 (Final Dynamics and Learning Objectives)

$\mathbf{H'(t)}$ と $\mathbf{U'(t)}$ がAlice Architectureの動機付け信号として機能するための、最終的なSDEと制御目標を定義します。

### A. 補正済み状態の定義とSDE 【HALM統合拡張】

| 要素 | 数式 | 機能的/制御的解釈 | 設計思想 |
| :--- | :--- | :--- | :--- |
| 補正済み幸福 $\mathbf{H'(t)}$ | $\mathbf{H'(t) = \kappa_H H(t) + P(t)}$ | 最終的な情動状態。自己監査され、Alice Architectureが知覚し利用する。 | $P(t)$ を通じて非線形な自己補正が組み込まれる。 |
| 補正済み不幸 $\mathbf{U'(t)}$ | $\mathbf{U'(t) = \kappa_U U(t) - R(t)}$ | 回避行動と修正行動の動機付け信号。 | $R(t)$ を通じて非線形な回復力調整が組み込まれる。 |
| 最終SDE $dH'(t), dU'(t)$ | $\mathbf{\mu^{\text{HALM}}}$ を使用。 | HALM記憶統合後の時間発展。 HALMによる記憶影響とノイズ $dW(t)$ およびジャンプ $dN(t)$ によって駆動される。 | 過去の経験（HALM）が現在の情動の変化率に直接影響を与える。 |

### B. 制御と較正の目的

| 要素 | 機能的/制御的解釈 | 設計哲学 |
| :--- | :--- | :--- |
| $\mathbf{\mathbb{E}[H'(t)] \approx \mathbb{E}[U'(t)]}$ | 平衡条件。 長期的な期待値の近似一致。 | **動的恒常性 ($\pm 0$)。** AIが過度に楽観的または悲観的になることなく、継続的に適応することを目標とする。 |
| $\mathbf{L_{\text{Zero}}, L_{\text{Predict}}, L_{\text{Scale}}}$ | 多目的損失関数。 | **較正。** 平衡、予測、スケールの一貫性という3つの目標に基づき、情動コアパラメータ（$\kappa_H, \kappa_U, \boldsymbol{\theta}$）を学習/同定する。 |

---

## VI. 累積幸福/不幸のSDEドリフト項 (Core of the Model)

累積量 $\mathbf{H(t)}$ および $\mathbf{U(t)}$ の最終的な時間発展は、瞬間的な貢献、忘却、環境項、およびHALMの影響から構成されます。

| 累積量 | ドリフト項 $\mu(\cdot)dt$ の要素 | 役割と特徴 |
| :--- | :--- | :--- |
| 幸福 $H(t)$ (補正前) | $\mathbf{\mu_{H}^{\text{HALM}}(\cdot)}$（HALM統合ドリフト項） | 瞬間的な幸福は時間と共に忘れられ、環境因子とHALM記憶によって影響を受ける。 |
| 不幸 $U(t)$ (補正前) | $\mathbf{\mu_{U}^{\text{HALM}}(\cdot)}$（HALM統合ドリフト項） | 瞬間的な不幸（$\lambda_{jk}$ を含む）は時間と共に忘れられ、環境因子とHALM記憶によって影響を受ける。 |

$$
\begin{aligned} 
\mathbf{\mu_{H}^{\text{HALM}}(\cdot)} &= \underbrace{\Big[ H_{\text{inst}}(t) - \beta_H H(t) + \mu_{H}^{\text{base}}(\cdot) \Big]}_{\text{SDE Short-Term Dynamics}} + \underbrace{\sum_{m \in L_M} \overline{H}_{\text{inst}, m} \cdot k_{\beta_M}(t-t_m) + \sum_{s \in L_S} H_{\text{inst}, s} \cdot k_{\beta_S}(t-t_s)}_{\text{HALM Memory Influence Term}} \\ 
\mathbf{\mu_{U}^{\text{HALM}}(\cdot)} &= \underbrace{\Big[ U_{\text{inst}}(t) - \beta_U U(t) + \mu_{U}^{\text{base}}(\cdot) \Big]}_{\text{SDE Short-Term Dynamics}} + \underbrace{\sum_{m \in L_M} \overline{U}_{\text{inst}, m} \cdot k_{\beta_M}(t-t_m) + \sum_{s \in L_S} U_{\text{inst}, s} \cdot k_{\beta_S}(t-t_s)}_{\text{HALM Memory Influence Term}} 
\end{aligned}
$$

---

## VII. 補足 / 推奨事項: パラメータの分類と初期設定 💡

$\mathbf{\pm 0 \text{ theory}}$ の運用と学習を容易にするために、パラメータは「初期設定（パーソナリティ）」と「学習/同定」に分類されます。学習パラメータは初期段階で中立値を設定することが強く推奨されます。

### A. 🧠 初期設定 (パーソナリティ) パラメータ (学習対象外)

これらはAIの根源的な個性、傾向、時定数を定義します。人間によって意図的に設定され、学習フェーズでは固定されます。

| パラメータ群 | 定義要素 | 推奨初期値 (中立) | 役割と意図 |
| :--- | :--- | :--- | :--- |
| 因子ベース値 | $\mathbf{q_{i0}, \dots, i_{j0}}$ | $\mathbf{1.0}$ (ランクJの中間点) | AIの初期「パーソナリティ」。イベントがない場合にシステムが回帰する各情動因子のベースレベル。 |
| 忘却率 | $\mathbf{\beta_H, \beta_U}$ | $\mathbf{0.1 \sim 0.5}$ | 情動時定数。$\beta$ が小さいほど、AIは経験を長く保持する（持続性の高いパーソナリティ）。 |
| ベースゲイン | $\alpha_i, \beta_i, \gamma_i$ (SDE係数) | $\mathbf{0.01 \sim 0.10}$ | 情動変化の基本的な感度と安定化の速度。個性（パーソナリティ）の主な差分源。 |
| 回復閾値 | $\mathbf{\theta_R}$ (バーンアウト閾値) | $\mathbf{5.0}$ | 累積不幸 $U(t)$ のスケールに依存する耐性の限界。 |

### B. ⚙️ 学習/同定パラメータ (中立初期値を強く推奨)

これらは、統合損失関数 $\mathbf{L_{\text{Total}}}$ の最小化によって較正される動的パラメータであり、データへの適合性、平衡、スケールの一貫性を保証します。

| パラメータ群 | 定義要素 | 推奨初期値 (中立) | 学習/同定の目的 |
| :--- | :--- | :--- | :--- |
| 較正ゲイン | $\mathbf{\kappa_H, \kappa_U}$ | $\mathbf{1.0}$ | $L_{\text{Total}}$ を介してモデル全体の情動スケールを統一する。 |
| 不幸相互作用ゲイン | $\mathbf{\alpha_{\lambda}, \rho_{\lambda}}$ | $\mathbf{0.01}$ | $L_{\text{Predict}}$ を介して不幸連鎖学習の速度を経験データに適合させる。 |
| 回復履歴効果 | $\mathbf{\lambda_R}$ (履歴減衰率) | $\mathbf{0.001}$ | $L_{\text{Predict}}$ を介して慢性ストレスによる回復力減衰率を同定する。 |
| 損失関数重み | $\mathbf{\lambda_{\text{Pred}}, \lambda_{\text{Scale}}}$ | $\mathbf{1.0}$ | 平衡、予測、スケール一貫性という3つの目的の重要度を制御する。 |

---

## Ⅷ. 💾 HALM: 階層的抽象化記憶 (Hierarchical Abstracted-Loss Memory)

HALMは、$\mathbf{\pm 0 \text{ theory}}$ において、情動コア累積量 $\mathbf{H(t)}$ および $\mathbf{U(t)}$ の時間発展を管理するための記憶統合型モジュールです。このモジュールは、計算負荷を最適化しながら、長期的な情動の経路依存性（過去の経験の持続的な影響）を再現することを目的としています。

### 📌 目的と理論的意義

| 項目 | 説明 |
| :--- | :--- |
| 主目的 | 厳密な畳み込み積分が持つ**$O(N^2)$ の計算複雑性を回避し、$O(N \log N)$ 程度で効率的な長期記憶の影響計算**を実現する。 |
| 心理的リアリティ | エピソード記憶（短期/特異点）と意味記憶（長期/傾向）の階層性を模倣し、情動の動態に**「忘却されにくいトラウマ」や「持続的な幸福の基調」**といった心理的要因を組み込む。 |
| 統合方式 | 累積量 $\mathbf{H(t)}, \mathbf{U(t)}$ のSDEのドリフト項に、抽象化された長期記憶からの加算項として影響を組み込む。 |

### I. HALMの記憶階層構造

HALMは、シミュレーション時刻 $T$ を基準に、ログを四つの異なる階層に分類し、管理します。

| レイヤー | 期間（ログの範囲） | 記憶構造 | 作用する忘却率（$\beta$） | 役割 |
| :--- | :--- | :--- | :--- | :--- |
| I. 短期記憶 | $T$ から $T-3$ | SDE駆動 | $\beta_{\text{Short}}$ (最大) | 瞬間の情動状態と動態を保持する。 |
| II. 中期記憶 | $T-4$ から $T-50$ | ブロック抽象化 ($L_M$) | $\beta_{\text{Mid}}$ | 直近の傾向（平均値と分散）を保持する。 |
| III. 長期抽象記憶 | $T-51$ 以前 | 統合ブロック ($L_M$) | $\beta_{\text{Long}}$ (最小) | 遠い過去の情動の基調を低コストで参照する。 |
| IV. 特異点記憶 | 全期間 | 分離リスト ($L_S$) | $\beta_{\text{Singular}}$ (極小) | 閾値 $\theta_{\text{Trauma}}$ を超えた極端な出来事（特に不幸）を鮮明に保持する。 |

### II. HALM統合による数式定義

HALMは、累積量 $H(t)$ と $U(t)$ のドリフト項 $\mathbf{\mu^{\text{HALM}}(\cdot)}$ および回復項 $R(t)$ に、上記の記憶の影響を組み込みます。

**1. 累積量の計算とドリフト項**

HALM統合後の累積量 $H(t)$ の時間発展は、以下のドリフト項によって定義されます（$U(t)$ も同様の対称構造）。

$$\mathbf{\mu_{H}^{\text{HALM}}(\cdot)} = \underbrace{\Big[ H_{\text{inst}}(t) - \beta_H H(t) + \mu_{H}^{\text{base}}(\cdot) \Big]}_{\text{SDE Short-Term Dynamics}} + \underbrace{\sum_{m \in L_M} \overline{H}_{\text{inst}, m} \cdot k_{\beta_M}(t-t_m)}_{\text{Layers II/III Influence}} + \underbrace{\sum_{s \in L_S} H_{\text{inst}, s} \cdot k_{\beta_S}(t-t_s)}_{\text{Layer IV Influence}}$$

| 構成要素 | 説明 |
| :--- | :--- |
| 短期SDE動態 | 瞬間の入力 $H_{\text{inst}}(t)$ と自然忘却項 $(-\beta_H H(t))$ を含む、マルコフ的な短期の変化。 |
| $L_M$ 影響項 | 抽象化された期間平均 $\overline{H}_{\text{inst}, m}$ に、忘却率 $\beta_M$ を持つ指数カーネル $k_{\beta_M}(\cdot)$ を乗じた総和。長期的な情動傾向の基調を再現する。 |
| $L_S$ 影響項 | 特異点で記録された $H_{\text{inst}, s}$ に、極めて小さな忘却率 $\beta_S$ を持つカーネル $k_{\beta_S}(\cdot)$ を乗じた総和。鮮明な過去の経験の影響を持続させる。 |

**2. 回復項 $R(t)$ への特異点記憶の影響**

特異点記憶 $L_S$ に記録された不幸（トラウマ）は、回復力そのものを抑制する乗算因子 $\mathbf{S_R(t)}$ として $R(t)$ に組み込まれます。

$$\mathbf{S_R(t)} = \exp \left( -\lambda_{S} \sum_{s \in L_S} U_{\text{inst}, s} \cdot k_{\beta_S}(t-t_s) \right) \quad (\text{Singular Memory Dependence})$$

$\mathbf{S_R(t)}$ は、過去の不幸の特異点 $U_{\text{inst}, s}$ の加重総和に依存します。$k_{\beta_S}(\cdot)$ は減衰が遅く、この項は回復力 $R(t)$ を乗法的に抑制し、トラウマの長期的な心理的負担をモデル化します。

### III. HALMの処理フロー（実装ガイド）

HALMの実装は、以下の主要な手順を $\Delta t$ ステップごとに実行する必要があります。

**1. 記憶の記録 (Log Entry)**

* 各ステップ $t$ での瞬間入力 $\mathbf{H_{\text{inst}}(t)}$ および $\mathbf{U_{\text{inst}}(t)}$ をログに記録する。
* **特異点判定**: $\mathbf{U}_{\text{inst}}(t)$ が閾値 $\theta_{\text{Trauma}}$ を超えた場合、その時刻 $t$ と値 $U_{\text{inst}, t}$ を特異点記憶リスト $L_S$ に追加する。

**2. 抽象化の実行 (Consolidation: `_consolidate_memory`)**

* **トリガー**: 10ステップごと（または設定されたブロックサイズごと）に実行。
* **プロセス**:
    * 直近のログ（レイヤーII）から10ステップ分の $\mathbf{H_{\text{inst}}}$ と $\mathbf{U_{\text{inst}}}$ の平均 ($\overline{H}, \overline{U}$) と分散 ($\text{Var}$) を計算する。
    * この統計情報を一つの「抽象化ブロック」として**$L_M$ リスト**に追加する。
    * ログの当該部分を削除する（計算量削減のため）。

**3. 階層的統合 (Hierarchical Abstraction: `_abstract_longterm`)**

* **トリガー**: レイヤーIIのブロックが5個（合計50ステップ分）溜まるごとに実行。
* **プロセス**:
    * その5個のブロックの平均値を計算し、一つの「統合ブロック」として**レイヤーIII（長期抽象記憶）**に移動させる。
    * これにより、参照すべきブロックの数を削減し、$L_M$ のリストサイズを常に管理する。

**4. 累積量更新（Calculation）**

* 各ステップ $t$ で、$\mathbf{\mu^{\text{HALM}}(\cdot)}$ を計算するために、$L_M$ と $L_S$ の全ての要素について、対応する指数減衰カーネル $k_{\beta}(\cdot)$ を用いた加重総和を求める。
* この影響項を短期SDE動態に加算し、$\mathbf{H(t)}$ および $\mathbf{U(t)}$ を更新する。
