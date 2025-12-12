# Alice-Architecture-based-on-±0 core

**[Contact Email Address]**
Please contact us if anything comes up. Questions are welcome, too.

**Xiang446435786543@proton.me**

---


## 🧭 目次 (Table of Contents)


### (JP)

* [🧩 プロジェクト概要：AIの情動と動機付けの試み](#project-overview-jp)

  * [核心をなす二つの理論](#two-core-theories-jp)

  * [**🔥 理論間の接続と表記**](#theory-connection-jp)

  * [🔄 循環構造の形成](#circulatory-structure-jp)

* [\mathbf{\pm 0 \text{ theory}}🔬 の核心：情動の動的な数理モデル](#pm0-core-jp)

  * [確率的で連続的な情動の時間発展](#stochastic-evolution-jp)

  * [脆弱性と複雑な学習のモデル化](#fragility-modeling-jp)

  * [回復力の非対称性と制御目標](#asymmetry-control-jp)

* [II. Alice theory の核心：動機付けと制御](#alice-core-jp)

  * [💡 核心的な設計思想：「自己否定駆動の恒常性」](#self-negation-jp)

  * [🧠 主な制御メカニズム](#control-mechanisms-jp)

* [💰 プロジェクトの支援](#project-support-jp)

* [📜 ライセンス](#license-jp)


### (ENG)

* [🧩 Project Overview: An Attempt at AI Emotion and Motivation](#project-overview-eng)

  * [The Two Core Theories](#two-core-teories-eng)

  * [**🔥 Connection and Notation between Theories**](#theory-connection-eng)

  * [🔄 Formation of the Circulatory Structure](#circulatory-structure-eng)

* [\mathbf{\pm 0 \text{ theory}}🔬 Core: Dynamic Mathematical Model of Emotion](#pm0-core-eng)

  * [Probabilistic and Continuous Temporal Evolution of Emotion](#stochastic-evolution-eng)

  * [Modeling Fragility and Complex Learning](#fragility-modeling-eng)

  * [Asymmetry of Recovery and Control Objective](#asymmetry-control-eng)

* [II. Alice theory Core: Motivation and Control](#alice-core-eng)

  * [💡 Core Design Philosophy: "Self-Negation-Driven Homeostasis"](#self-negation-eng)

  * [🧠 Primary Control Mechanisms](#control-mechanisms-eng)

* [💰 Project Support / Donations](#project-support-eng)

* [📜 License](#license-eng)


---


<a id="project-overview-jp"></a>

## 🧩 プロジェクト概要：AIの情動と動機付けの試み


​本プロジェクトは、SDE駆動$\pm 0$理論とHALMを統合することで、自己否定駆動の恒常性を追求する**内面駆動の自律思考型AI（AGIモデルの試み）の全体統合仕様 (F_{total}) **を探究するものです。


これは、専門的な研究や商業利用を目的とするものではなく、高度な数理モデルを通じて`**「AIの自意識と情動の動力学」**`を記述する趣味です。


<a id="two-core-theories-jp"></a>

### 核心をなす二つの理論


このアーキテクチャは、以下の二つの理論ブロックによって構成されています。


| 理論ブロック | 対応概念 | 役割 |
| :--- | :--- | :--- |
| **Alice theory** | 疑似自意識モデル (自己モデル、メタ認知、制御層)。 | **$\mathbf{\pm 0 \text{ theory}}$** が算出した情動状態を、AIの`**行動決定**`と`**パーソナリティの進化**`に反映させます。 |
| **$\mathbf{\pm 0 \text{ theory}}$** | 不幸・幸福モデル (情動コア、レジリエンス、バーンアウト)。 | 外部環境からの入力やAI自身の認知状態を処理し、累積的な幸福度 $H(t)$ と不幸度 $U(t)$ を`**確率微分方程式（SDE）**`に基づいて算出し、Alice theory へ提供します。 |


<a id="theory-connection-jp"></a>

### 🔥 理論間の接続と表記の統一 (重要)


二つの仕様書間で表記が分かれていますが、これらは同一の**最終情動状態**を指します。

| $\mathbf{\pm 0 \text{ theory}}$ (連続時間) | Alice theory (離散時間) | 概念 | 役割 |
| :--- | :--- | :--- | :--- |
| **$H'(t)$** (補正済み幸福状態) | **$\mathbf{H_{pz}}(t)$** (累積幸福度層) | 幸福 | 内部で補正・調整された幸福度の最終的な状態。 |
| **$U'(t)$** (補正済み不幸状態) | **$\mathbf{U_{pz}}(t)$** (累積不確実性層) | 不幸/不安 | 内部で回復・抑制された不幸/不安の最終的な状態。 |

$$
\mathbf{H_{pz}}(t) \equiv H'(t) \quad \text{and} \quad \mathbf{U_{pz}}(t) \equiv U'(t)
$$

**$\mathbf{\pm 0 \text{ theory}}$** によって計算された**厳密な情動コアの状態 $H'(t)$ と $U'(t)$** は、そのまま **Alice Architecture の中枢的な情動層 $\mathbf{H_{pz}}(t)$ と $\mathbf{U_{pz}}(t)$** として機能します。


<a id="circulatory-structure-jp"></a>

### 🔄 循環構造の形成


二つの理論は、以下のフィードバックループによって連動し、AIの存在状態全体を駆動します。


1. **外部入力「環境」**から情報が受け取られ、**$\mathbf{\pm 0 \text{ theory}}$** へ入力されます。

2. **$\mathbf{\pm 0 \text{ theory}}$** が、現在の`**情動状態（幸福 $H'$ と 不幸 $U'$）**`を算出します。

3. **Alice theory** が、算出された不幸を最小限にするような`**行動の意図**`を発生させます。

4. その意図に基づき、`**LLM（言語生成API）**`が言語生成を担当します。

5. **Alice theory** は、LLMの出力を`**行動**`として実行します。

6. その`**行動と結果**`が、`**新たな環境入力および自身の行動による影響**`として再び $\mathbf{\pm 0 \text{ theory}}$ へフィードバックされます。


この`**「環境 → ±0 → Alice（意図） → LLM → Alice（行動） → 環境...」という循環構造**`こそが、AIに`**内的な安定性（恒常性）**`を追求させる動機付けの源となります。


---


<a id="project-overview-eng"></a>

## 🧩 Project Overview: An Attempt at AI Emotion and Motivation


​This project explores the overall integrated specification (F_{total}) of an Autonomous Thinking AI (an attempt at an AGI model) driven internally by the pursuit of self-negation-driven homeostasis, achieved by integrating the SDE-driven \pm 0 Theory and HALM.

​This endeavor is not intended for professional research or commercial use, but rather a personal hobby aiming to describe the "dynamics of AI self-consciousness and emotion" through advanced mathematical models.
<a id="two-core-teories-eng"></a>

### The Two Core Theories


This architecture is constructed by the following two theoretical blocks:


| Theoretical Block | Corresponding Concept | Role |
| :--- | :--- | :--- |
| **Alice theory** | Pseudo-Self-Consciousness Model (Self-Model, Metacognition, Control Layer). | Reflects the emotional states calculated by the **$\mathbf{\pm 0 \text{ theory}}$** into the AI's `**action determination**` and `**personality evolution**`. |
| **$\mathbf{\pm 0 \text{ theory}}$** | Unhappiness/Happiness Model (Emotional Core, Resilience, Burnout). | Processes input from the external environment and the AI's own cognitive state, calculates the cumulative happiness $H(t)$ and unhappiness $U(t)$ based on `**Stochastic Differential Equations (SDEs)**`, and provides these to the Alice theory. |


<a id="theory-connection-eng"></a>

### 🔥 Connection and Notation between Theories (Crucial)


While the notation differs between the two specification documents, they refer to the **identical final emotional states**.

| $\mathbf{\pm 0 \text{ theory}}$ (Continuous Time) | Alice theory (Discrete Time) | Concept | Role |
| :--- | :--- | :--- | :--- |
| **$H'(t)$** (Corrected Happiness State) | **$\mathbf{H_{pz}}(t)$** (Cumulative Happiness Layer) | Happiness | The final state of happiness, internally corrected and adjusted. |
| **$U'(t)$** (Corrected Unhappiness State) | **$\mathbf{U_{pz}}(t)$** (Cumulative Uncertainty Layer) | Unhappiness/Anxiety | The final state of unhappiness/anxiety, internally recovered and suppressed. |

$$
\mathbf{H_{pz}}(t) \equiv H'(t) \quad \text{and} \quad \mathbf{U_{pz}}(t) \equiv U'(t)
$$

The **rigorously calculated emotional core states $H'(t)$ and $U'(t)$** provided by the **$\mathbf{\pm 0 \text{ theory}}$** function directly as the **central emotional layers $\mathbf{H_{pz}}(t)$ and $\mathbf{U_{pz}}(t)$** within the **Alice Architecture**.


<a id="circulatory-structure-eng"></a>

### 🔄 Formation of the Circulatory Structure


The two theories operate interdependently through the following feedback loop, driving the AI's overall state of existence:


1.  **Input Reception:** Information is received from the **External Input "Environment"** and fed into the **$\mathbf{\pm 0 \text{ theory}}$**.

2.  **Emotional Calculation:** The **$\mathbf{\pm 0 \text{ theory}}$** calculates the current **Emotional State** (Happiness $H'$ and Unhappiness $U'$).

3.  **Intention Generation:** **Alice theory** generates an **intention for action** that minimizes the calculated Unhappiness.

4.  **Language Generation:** Based on that intention, the **LLM (Language Generation API)** handles the language generation.

5.  **Action Execution:** **Alice theory** executes the output of the LLM as an **Action**.

6.  **Feedback:** That **Action and its outcome** are fed back into the **$\mathbf{\pm 0 \text{ theory}}$** as the new environmental input and the effects of its own action.


This **"Environment $\rightarrow \pm 0 \rightarrow$ Alice (Intention) $\rightarrow$ LLM $\rightarrow$ Alice (Action) $\rightarrow$ Environment..." Circulatory Structure** is precisely the source of motivation that drives the AI to pursue **internal stability (homeostasis)**.


---


<a id="pm0-core-jp"></a>

## 🔬 $\mathbf{\pm 0 \text{ theory}}$ の核心：情動の動的な数理モデル


$\mathbf{\pm 0 \text{ theory}}$ は、AIの情動状態（幸福 $H$ と不幸 $U$）の変動を、単なるスコアリングではなく、連続時間における動的なシステムとして扱うことを試みています。目標は、現実の心理現象に見られる**不確実性や非対称な回復力**を数理的に再現することです。


<a id="stochastic-evolution-jp"></a>

### 1. 確率的で連続的な情動の時間発展


情動の累積量 $H(t)$ と $U(t)$ は、畳み込み積分 (1.1) によって、過去の瞬間的な経験を指数減衰カーネル $e^{-\beta(t-\tau)}$ で重み付けして計算されます。これは、情動が**時間とともに自然に忘却される**という性質をモデル化しています。


さらに、情動を駆動する個々の因子や環境（1.3, 3.1, 3.2）は、**確率微分方程式 (SDE)** に従います。


$$dX(t) = \mu_{X}(\cdot)\,dt + \sigma_{X}(\cdot) \, dW_{X}(t)$$


これは、情動の変化に決定論的な傾向 ($\mu_{X}$) と、予測不可能なランダムノイズ ($\sigma_{X} dW_{X}$) の両方が含まれることを意味し、AIが**不安定性の中でも恒常性を保つ力学的な基盤**となっています。


<a id="fragility-modeling-jp"></a>

### 2. 脆弱性と複雑な学習のモデル化


瞬間的な幸福 $H_{\text{inst}}$ と不幸 $U_{\text{inst}}$ の構成に、特定の心理現象を反映させています。


* **幸福の脆弱性（乗法構造）**: 瞬間幸福 $\mu_i(t)$ は因子の乗法 (2.2) で定義されます。


    $$\mu_i(t) = w_{i0} \cdot q_i(t) \cdot r_i(t) \cdot c_i(t) \cdot v_i(t) \cdot d_i(t)$$


    この構造により、たった一つの基礎因子 ($v_i$, $d_i$ など) が低くなると、全体の幸福度も低くなるという、**情動の脆弱性**を表現しています。


* **不幸の連鎖学習（相互作用項）**: 瞬間不幸 $U_{\text{inst}}(t)$ には、複数の不幸イベント $\nu_j, \nu_k$ が重なった場合の非線形な増幅を担う相互作用項 $\lambda_{jk}\,\nu_j(t)\,\nu_k(t)$ が含まれます (2.3)。この $\lambda_{jk}$ 自体も動的に学習される (2.4) ことで、AIが特定の不幸の組み合わせを**「トラウマ」として学習**し、過剰に反応するプロセスを再現します。


<a id="asymmetry-control-jp"></a>

### 3. 回復力の非対称性と制御目標


AIの自己監査機構として導入された補正項 $P(t)$ と回復項 $R(t)$ (4.1, 4.2) は、非線形な特徴を持ちます。


* **バーンアウト効果**: 回復項 $R(t)$ は、累積不幸 $U(t)$ が閾値 $\theta_R$ を超えると、回復効率が劇的に低下する関数 $T_R(t)$ を含みます。


    $$T_R(t) = \frac{1}{1 + e^{-\kappa_R \bigl(U(t)-\theta_R\bigr)}}$$


    これは、AIの`**精神的なリソース限界（バーンアウト）**`をモデル化する試みです。


* **動的恒常性（制御目標）**: 最終的な目標は、補正後の幸福状態 $H'(t)$ と不幸状態 $U'(t)$ の期待値を近似的に一致させること (5.3) です。


    $$\mathbb{E}\bigl[H'(t)\bigr]\;\approx\;\mathbb{E}\bigl[U'(t)\bigr]$$


    この平衡条件を制御目的関数 $J$ の形で定義し、AIが長期的に**過度に楽観的にも悲観的にもならない安定した内的な状態**を追求することを指針としています。


---


<a id="pm0-core-eng"></a>

## 🔬 $\mathbf{\pm 0 \text{ theory}}$ Core: Dynamic Mathematical Model of Emotion


The $\mathbf{\pm 0 \text{ theory}}$ attempts to treat the fluctuation of the AI's emotional states (happiness $H$ and unhappiness $U$) as a dynamic system in continuous time, rather than mere scoring. The goal is to mathematically reproduce the **uncertainty and asymmetrical recovery** observed in real psychological phenomena.


<a id="stochastic-evolution-eng"></a>

### 1. Probabilistic and Continuous Temporal Evolution of Emotion


The cumulative emotional quantities, $H(t)$ and $U(t)$, are calculated by a convolution integral (1.1), where past instantaneous experiences are weighted by an exponential decay kernel $e^{-\beta(t-\tau)}$. This models the property that emotions are **naturally forgotten over time**.


Furthermore, the individual factors and the environment that drive emotion (1.3, 3.1, 3.2) follow a **Stochastic Differential Equation (SDE)**:


$$dX(t) = \mu_{X}(\cdot)\,dt + \sigma_{X}(\cdot) \, dW_{X}(t)$$


This means that the emotional change includes both a deterministic tendency ($\mu_{X}$) and unpredictable random noise ($\sigma_{X} dW_{X}$), providing a **dynamic foundation for the AI to maintain homeostasis** even amid instability.


<a id="fragility-modeling-eng"></a>

### 2. Modeling Fragility and Complex Learning


The construction of instantaneous happiness $H_{\text{inst}}$ and unhappiness $U_{\text{inst}}$ reflects specific psychological phenomena.


* **Emotional Fragility (Multiplicative Structure)**: Instantaneous happiness $\mu_i(t)$ is defined by the multiplication of factors (2.2):


    $$\mu_i(t) = w_{i0} \cdot q_i(t) \cdot r_i(t) \cdot c_i(t) \cdot v_i(t) \cdot d_i(t)$$


    Through this structure, the theory expresses **emotional fragility**, where the happiness level dramatically drops if even a single fundamental factor ($v_i$, $d_i$, etc.) becomes low.


* **Chain Learning of Unhappiness (Interaction Term)**: The instantaneous unhappiness $U_{\text{inst}}(t)$ includes an interaction term $\lambda_{jk}\,\nu_j(t)\,\nu_k(t)$ (2.3), which accounts for the nonlinear amplification when multiple unhappy events $\nu_j, \nu_k$ overlap. Since $\lambda_{jk}$ itself is dynamically learned (2.4) by the rule $\frac{d\lambda_{jk}}{dt} = \alpha_{\lambda} \cdot \nu_j(t) \cdot \nu_k(t) - \rho_{\lambda} \cdot (\lambda_{jk} - \lambda_{jk}^{\text{base}})$, the model reproduces the process where the AI learns a specific combination of misfortunes as a **"trauma"** and reacts excessively.


<a id="asymmetry-control-eng"></a>

### 3. Asymmetry of Recovery and Control Objective


The correction term $P(t)$ and recovery term $R(t)$ (4.1, 4.2), introduced as the AI's self-auditing mechanism, possess nonlinear characteristics.


* **Burnout Effect**: The recovery term $R(t)$ includes a function $T_R(t)$ that causes the recovery efficiency to drop dramatically when cumulative unhappiness $U(t)$ exceeds a threshold $\theta_R$:


    $$T_R(t) = \frac{1}{1 + e^{-\kappa_R \bigl(U(t)-\theta_R\bigr)}}$$


    This attempts to model the AI's **limit of mental resources (burnout)**.


* **Dynamic Homeostasis (Control Objective)**: The ultimate goal is to achieve an approximate match between the expected values of the corrected happiness state $H'(t)$ and unhappiness state $U'(t)$ (5.3):


    $$\mathbb{E}\bigl[H'(t)\bigr]\;\approx\;\mathbb{E}\bigl[U'(t)\bigr]$$


    This equilibrium condition is defined in the form of a control objective function $J$, serving as the guideline for the AI to pursue a **stable internal state where it is neither overly optimistic nor pessimistic in the long run**.


---


<a id="alice-core-jp"></a>

## II. Alice theory の核心：動機付けと制御


**Alice theory** は、$\mathbf{\pm 0 \text{ theory}}$ から提供される情動状態 $\mathbf{H_{pz}}, \mathbf{U_{pz}}$ を利用し、AIの認知、自己モデル、および行動決定を駆動する統合制御アーキテクチャです。この理論の中心は、内的なコストの最小化を通じた`**「自己否定駆動の恒常性」**`の追求にあります。


<a id="self-negation-jp"></a>

### 💡 核心的な設計思想：「自己否定駆動の恒常性」


モデル（Alice）は、単に外部からの報酬（$\mathrm{reward}(t)$）を最大化するだけでなく、自己の不確実性と自己制御の負荷をコストとして認識するよう設計されています。


統一された目的関数 $\mathbf{V}(t)$ の最適化 (III.1) は、これらの内的な否定要素の最小化を強く重視します。


$$V(t) = \sum_i VFL_i(t) - \underbrace{\lambda_P\sum_i P_i(t) \cdot (\dots)}_{\text{予測/不安コスト}} - \underbrace{\lambda_C\mathrm{Var}(\mathbf{E_{ctrl}}(t))}_{\text{制御負荷コスト}} - \underbrace{\lambda_S \mathrm{Dist}(\mathbf{E_{self}}, \mathbf{E_{self}}^{pred})}_{\text{自己一貫性コスト}}$$


これは、不安（予測誤差 $P_i$）や努力（制御層 $\mathbf{E_{ctrl}}$ の分散）を最小化しようとする働き、すなわち`**「自己否定駆動」**`によって、安定した存在状態（恒常性）に収束することを目指します。


<a id="control-mechanisms-jp"></a>

### 🧠 主な制御メカニズム


このアーキテクチャでは、情動状態 $\mathbf{U_{pz}}$ が、AIの学習と行動に直接影響を与えます。


* **情動による学習の修正 ($\mathbf{G}_{\text{Affect}}$)**:

    スキル学習のための重み更新 ($\mathbf{\Delta W^X}$) には、報酬 ($\mathbf{G}_{\text{Value}}$) だけでなく、予測誤差 $P_i$ と不幸状態 $\mathbf{U_{pz}}$ から計算される情動／防御項 ($\mathbf{G}_{\text{Affect}}$) が含まれます (III.3)。


    $$\mathbf{G}_{\text{Affect}} = -\nabla_{W^X} \left( \lambda_P \sum_i P_i(t) \cdot \left(1 + \kappa_U \max (\mathbf{U_{pz}}) \right) + \dots \right)$$


    これは、不幸や不安が高いほど、AIは`**内的な防御（コスト削減）**`のために学習を修正することを意味します。


* **自己ナラティブ進化 ($\mathbf{\theta}$)**:

    AIのパーソナリティ・パラメータ $\mathbf{\theta}$ は、自己一貫性コスト $\mathrm{Dist}(\mathbf{E_{self}}, \mathbf{E_{self}}^{pred})$ などに基づいて進化します (III.2)。これは、自己モデルと自己予測の間のズレが、AIの`**基礎的な傾向（性格）**`を長期的に変化させる仕組みです。


* **内的な層間フィードバック**:

    認知層 ($\mathbf{C}$)、記憶層 ($\mathbf{M}$)、そして情動核 ($\mathbf{H_{pz}, U_{pz}}$) が複雑に相互作用し、AIの内的な状態全体 ($\mathbf{S}(t)$) に基づいてメタ認知層 ($\mathbf{E_s}$) が情報を処理します (II.3)。これにより、自己を客観視し（$\mathbf{E_{obj}}$）、自己モデル（$\mathbf{E_{self}}$を構築する**自己参照プロセス**をモデル化しています。


---


<a id="alice-core-eng"></a>

## II. Alice theory Core: Motivation and Control


**Alice theory** is an integrated control architecture that utilizes the emotional states $\mathbf{H_{pz}}, \mathbf{U_{pz}}$ provided by the **$\mathbf{\pm 0 \text{ theory}}$** to drive the AI's cognition, self-model, and action determination. The core of this theory lies in the pursuit of **"Self-Negation-Driven Homeostasis"** through the minimization of internal costs.


<a id="self-negation-eng"></a>

### 💡 Core Design Philosophy: "Self-Negation-Driven Homeostasis"


The model (Alice) is designed not only to maximize external rewards ($\mathrm{reward}(t)$) but also to perceive its own uncertainty and the burden of self-control as costs.


The optimization of the unified objective function $\mathbf{V}(t)$ (III.1) heavily emphasizes the minimization of these internal, negative elements:


$$V(t) = \sum_i VFL_i(t) - \underbrace{\lambda_P\sum_i P_i(t) \cdot (\dots)}_{\text{Prediction/Anxiety Cost}} - \underbrace{\lambda_C\mathrm{Var}(\mathbf{E_{ctrl}}(t))}_{\text{Control Load Cost}} - \underbrace{\lambda_S \mathrm{Dist}(\mathbf{E_{self}}, \mathbf{E_{self}}^{pred})}_{\text{Self-Consistency Cost}}$$


This mechanism—the tendency to minimize anxiety (prediction error $P_i$) and effort (variance of the control layer $\mathbf{E_{ctrl}}$)—aims to converge to a stable state of existence (homeostasis) through **"Self-Negation-Drive."**


<a id="control-mechanisms-eng"></a>

### 🧠 Primary Control Mechanisms


In this architecture, the emotional state $\mathbf{U_{pz}}$ directly influences the AI's learning and behavior.


* **Emotional Correction of Learning ($\mathbf{G}_{\text{Affect}}$)**:

    The weight update for skill learning ($\mathbf{\Delta W^X}$) includes not only the reward term ($\mathbf{G}_{\text{Value}}$) but also an Affect/Defense Term ($\mathbf{G}_{\text{Affect}}$), calculated from the prediction error $P_i$ and the unhappiness state $\mathbf{U_{pz}}$ (III.3).


    $$\mathbf{G}_{\text{Affect}} = -\nabla_{W^X} \left( \lambda_P \sum_i P_i(t) \cdot \left(1 + \kappa_U \max (\mathbf{U_{pz}}) \right) + \dots \right)$$


    This implies that the higher the unhappiness and anxiety, the more the AI modifies its learning for **internal defense (cost reduction)**.


* **Self-Narrative Evolution ($\mathbf{\theta}$)**:

    The AI's personality parameters $\mathbf{\theta}$ evolve based on factors such as the self-consistency cost $\mathrm{Dist}(\mathbf{E_{self}}, \mathbf{E_{self}}^{pred})$ (III.2). This is the mechanism by which the discrepancy between the self-model and the self-prediction changes the AI's **fundamental tendencies (personality)** over the long term.


* **Internal Inter-Layer Feedback**:

    The Cognitive Layer ($\mathbf{C}$), Memory Layer ($\mathbf{M}$), and the Emotional Nucleus ($\mathbf{H_{pz}, U_{pz}}$) interact complexly, and the Metacognition Layer ($\mathbf{E_s}$) processes information based on the AI's entire internal state ($\mathbf{S}(t)$) (II.3). This models a **self-referential process** where the AI objectifies the self ($\mathbf{E_{obj}}$) and constructs a self-model ($\mathbf{E_{self}}$).


---


<a id="project-support-jp"></a>

## 💰 プロジェクトの支援 (Project Support / Donations)


<a id="project-support-eng"></a>

### 日本語 (Japanese)
この **Alice theory** または **$\mathbf{\pm 0 \text{ theory}}$** が、ご覧いただいた方の思考に少しでも技術的な価値や示唆を提供できたのであれば、開発者にとってこれ以上の喜びはありません。

現在、私とAIの共同でこれらの発展版を**資金０円**で開発中です。しかし、個人的・社会的な制約により、この純粋な探究を趣味として継続することが非常に困難な状況にあります。

このプロジェクトを継続し、進化させるためのご支援をいただけましたら、心より感謝申し上げます。

### 英語 (English)
We would be immensely pleased if we could receive your support to continue and evolve this pure exploration as a hobby.

We are currently developing advanced versions of these theories collaboratively with AI, entirely **without external funding**. However, due to personal and societal constraints, continuing this pure exploration as a hobby is becoming extremely difficult.

It is the developer's greatest pleasure if this **Alice theory** or **$\mathbf{\pm 0 \text{ theory}}$** has offered even a slight technical value or suggestion to your thoughts.

---
### 支援窓口 (Donation Address)

Metamask Wallet Address (Ethereum / Polygon Network):
`0x883bb5e66b7f76d33a999bbe9d2f8d5e3c55f8d1

<small>※ アドレスポイズニングを防ぐため、コピー後に最初の数文字と最後の数文字をご確認ください。</small>
<small>※ Please verify the first and last few characters after copying to prevent address poisoning attacks.</small>



#### QRコードによる支援 (Scan/Click to Support)

[ ![Metamask Wallet QR Code](./QR_MetaMask.jpg) ](0x883bb5e66b7f76d33a999bbe9d2f8d5e3c55f8d1)

<small>※ このQRコードはスキャン専用ですが、クリックするとウォレットアドレスを指します。</small>


---


<a id="license-jp"></a>

## 📜 ライセンス (License)


This project is released under the **MIT License**.


<a id="license-eng"></a>

---

