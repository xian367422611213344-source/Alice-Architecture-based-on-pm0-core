# 📘 Alice Architecture: Integrated Logic Final Specification for Emotion, Personality, and Intelligence — FORMULA_MAPPING

The purpose of this document is to completely clarify the self-stabilization mechanism of the Alice Architecture system by clearly mapping all core mathematical model elements to their functional roles, evolutionary rules, and underlying design philosophy.

---

<a id="objective-function"></a>
## I. 👑 Overall Objective Function and Self-Driven Motivation (Goal: Maximize $V(t)$)

The objective function $V(t)$ is defined by maximizing the external reward $\mathbf{VFL}$ and minimizing three internal costs (the self-negation-driven core).

$$
V(t) = \sum_{\mathbf{i}} \mathbf{VFL}_i(t) - L_{\text{P}}(t) - L_{\text{C}}(t) - L_{\text{S}}(t)
$$

| Formulaic Element | Logical Role (Mathematical) | Cognitive Meaning (Function) | Design Rationale (Why Necessary) |
| :--- | :--- | :--- | :--- |
| $\mathbf{S}(t)$ | The concatenated vector of all 14 layers. | The instantaneous definition of the **"self"** (the subject of introspection). Excludes $\mathbf{A}(t)$. | Separates the output of thought from the core existence of the self, establishing the foundation for introspection. |
| $\mathbf{b}^C \sim \mathbf{0}$ | Zero initialization of the bias term for the $\mathbf{C}$ layer. | Exclusion of initial **"prejudice."** | Prevents specific neurons from being unfairly activated during early learning, ensuring pure information transmission. |
| $-\lambda_P\sum P_i(t) \cdot (\dots)$ | Negative weight on the sum of prediction error $\mathbf{P}(t)$ (including anxiety amplification). | **"Anxiety"** or **"Unpredictability Cost."** | A self-negation-driven core that compels the model to seek a predictable, stable state. Most emphasized with $\lambda_P=1.0$. |
| $-\lambda_C\mathrm{Var}(\mathbf{E_{ctrl}}(t))$ | Negative weight on the variance of the Control Layer's fluctuation. | **"Effort/Load"** or **"Control Cost."** | Makes a state of easy control (stability of consciousness) a condition for maximizing $V(t)$. $\lambda_C=0.5$. |
| $\kappa_U(\theta)$ | Anxiety amplification coefficient (personality-dependent). | A **panic amplifier** when $\mathbf{U}_{pz}$ is high. | Incorporates strong risk aversion into behavioral choice by non-linearly amplifying prediction error during an anxious state. |
| $\mathrm{Dist}(\mathbf{E_{self}}, \mathbf{E_{self}}^{pred})$ | Distance cost between the predicted and observed self-model. | **"Discomfort due to Lack of Self-Consistency."** | Makes the coherence of the self-model a prerequisite for learning, promoting the stabilization of identity. $\lambda_S=0.8$. |

---

<a id="emotional-nucleus"></a>
## II. ❤️ Definition of the Emotional Nucleus ($\mathbf{\pm 0 \text{ theory}}$ Dynamics) and Personality ($\mathbf{\theta}$)

The emotional nucleus ($\mathbf{H_{pz}}, \mathbf{U_{pz}}$) is driven by the personality parameters $\mathbf{\theta}$, which themselves evolve based on the self's experience.

### A. Emotional Core Dynamics

The emotional core update is done via SDE discretization approximation (`ZeroOneTheory`).

$$
\mathbf{H}_{pz}(t+1) = \max \left(0, (1-\beta_H) \mathbf{H}_{pz}(t) + \alpha_H \mathbf{R}(t) - \gamma_{HU} \mathbf{U}_{pz}(t) + \epsilon_H \right)
$$

$$
\mathbf{U}_{pz}(t+1) = \max \left(0, (1-\beta_U) \mathbf{U}_{pz}(t) + \alpha_U \mathbf{P}(t) - \gamma_{UH} \mathbf{H}_{pz}(t) + \epsilon_U \right)
$$

| Formulaic Element | Logical Role (Mathematical) | Cognitive Meaning (Function) | Design Rationale |
| :--- | :--- | :--- | :--- |
| $\mathbf{H}_{pz}(t)$, $\mathbf{U}_{pz}(t)$ | Cumulative Happiness Layer, Cumulative Uncertainty Layer. | Long-term **"Sense of Well-being"** and **"Vigilance."** | Holds sustained satisfaction/risk assessment as an internal state, not just instantaneous reward/error. |
| $(1 - \beta_H(\theta))$, $(1 - \beta_U(\theta))$ | Decay terms dependent on the forgetting rate $\beta$. | **"Personality-Based Forgetting Speed."** | Implements personality traits such as optimistic/pessimistic via $\mathbf{\theta}$. Initial value $\mathbf{0.1}$. |
| $\gamma_{HU}(\theta)$, $\gamma_{UH}(\theta)$ | Mutual inhibition terms (Initial value $\mathbf{0.5}$). | **"Emotional Balance Control."** | Prevents happiness and anxiety from peaking simultaneously, maintaining a clear emotional polarity. |
| $\sum_{i} R_i(t)$, $\sum_{i} P_i(t)$ | Sum of Reward Layer $\mathbf{R}$, Sum of Prediction Error Layer $\mathbf{P}$. | Current **positive/negative input.** | $\mathbf{H}_{pz}$ is driven by reward, and $\mathbf{U}_{pz}$ is driven by uncertainty. |

### B. 💫 Personality Parameter ($\mathbf{\theta}$) Evolutionary Rules (Update: $\mathbf{\theta}_{t+1}=\mathbf{\theta}_t + \alpha(\dots)$)

| Evolutionary Signal | Driving Source | Target ($\theta$ Element) | Direction of Action | Effect (Meaning) |
| :--- | :--- | :--- | :--- | :--- |
| $\mathbf{\Delta SNEL}$ (Self-Narrative Evolution) | Self-discrepancy $\mathrm{Dist}(\cdot)$ and Will Stabilization $f_{\text{Will}}(\cdot)$. | Uncertainty forgetting rate $\beta_U$ | Increase | Decreases the persistence of anxiety, improving resilience. |
| | | Anxiety amplification coefficient $\kappa_U$ | Decrease | Suppresses the subjective amplification of objective anxiety. |
| $\mathbf{\Delta ISL}$ (Willpower Evolution) | Low control load $\mathrm{Var}(\mathbf{E_{ctrl}})$ and high internal satisfaction $\frac{V}{\sum VFL}$. | Happiness forgetting rate $\beta_H$ | Decrease | Increases the persistence of positive experiences. |
| | | Happiness set-point $H_{\text{base}}$ | Increase | Raises the baseline of wellbeing. |
| $f_{\text{Will}}(\cdot)$ | $\frac{1}{1 + \mathrm{Var}(\mathbf{H}(t))} \cdot \mathbb{E}[\mathbf{R}(t)]$ | Multiplier for $\mathbf{\Delta SNEL}$. | N/A | Strengthens willpower with stable behavior ($\mathrm{Var}(\mathbf{H})$ low) and high expected reward, stabilizing the consistency of behavior (personality). |

---

<a id="cognitive-layer"></a>
## III. 🧠 Structure and Dynamics of the Intelligence/Cognitive Layer

### A. Definition of Environmental Input $\mathbf{E_{env}}$

$\mathbf{E_{env}}$ is the input to the $\mathbf{C}$ layer and acts as the initial trigger for $\mathbf{U}_{pz}$ as a noisy representation of reality.

| Sub-Vector | Dimension | Role and Components | Design Rationale |
| :--- | :--- | :--- | :--- |
| $\mathbf{E}_{\text{Token}}$ | 512 | Semantic abstraction of language. | Adopts standard LLM dimension for handling primary information transmission. |
| $\mathbf{E}_{\text{Context}}$ | 128 | Encoding of non-linguistic meta-information. | Provides situational metadata like task structure and action history to $\mathbf{C}$. |
| $\mathbf{E}_{\text{Scalar}}$ | 4 | Immediate scalar information. | Handles $\mathrm{reward}(t)$, time $\mathrm{t}$, elapsed time $\mathrm{t}_{\text{elapsed}}$, and change flag $\mathrm{IsChange}$, acting as a primary input to $\mathbf{P}$. |
| Total | $\mathbf{n_{E_{env}}} = 644$ | N/A | Basis for the internal layer dimension design. |

### B. Cognitive Layer Update Rules

| Formulaic Element | Role and Structure | Cognitive Meaning (Function) | Design Rationale |
| :--- | :--- | :--- | :--- |
| $\mathbf{C}(t+1) = f_C(\dots)$ | RNN-form self-recurrence. | **"Continuity of Thought"** and **"Transparency of Information Source Weighting."** | Constructs current meaning from past context, the external world ($\mathbf{E_{env}}$), and memory ($\mathbf{M}$). |
| $(1-\alpha_M) \mathbf{M}(t)$ | Exponential decay term for Memory Layer $\mathbf{M}$. | **"Natural Forgetting."** | Prevents memory capacity saturation and maintains adaptability to new information. |
| $U^{C\leftarrow RRL} \mathbf{RRL}(t)$ | Connection term from $\mathbf{RRL}$ to $\mathbf{C}$. | **"Prediction from Structured Knowledge."** | Generates short-term predictions from long-term patterns, enabling $\mathbf{P}(t)$ calculation. |

---

<a id="learning-rule"></a>
## IV. 🛠️ Skill Learning Rule (Affective-TDL) and Stabilization Strategy

### A. Affective Temporal Difference Learning ($\text{A-TDL}$)

The update of weights $\mathbf{W}^X$ is driven directly by Value, Affect, and Coherence.

| Learning Term | Driving Source of Gradient | Role (Meaning) | Design Rationale |
| :--- | :--- | :--- | :--- |
| 🥇 $\mathbf{G}_{\text{Value}}$ | Maximization of long-term wellbeing $V$. | Optimization of Knowledge and Skills (Traditional RL). $\gamma=0.99$ emphasizes future value. |
| 🥈 $\mathbf{G}_{\text{Affect}}$ | Minimization of prediction error ($\lambda_P$) and control load ($\lambda_C$). | Learning to Avoid Mental Distress (**Self-Defense**). | Avoids anxiety and fatigue, embedding emotionally stable behavior as a skill. |
| 🥉 $\mathbf{G}_{\text{Coherence}}$ | Minimization of self-discrepancy $\mathrm{Dist}(\mathbf{E_{self}}, \mathbf{E_{self}}^{pred})$. | Maintenance of Behavioral Consistency with **Self-Model**. | Fixes behaviors that align with the self-narrative ("what I am"). |

### B. 🌪️ Hierarchical Strategy for the Noise Term ($\mathbf{\epsilon}$)

To control the trade-off between stability and exploration, noise variance is defined per layer and decays over time.

| Applied Layer X | Initial Variance $\sigma_{X,0}^2$ | Role and Design Rationale |
| :--- | :--- | :--- |
| $\mathbf{A}$ (Action) | 0.20 | Exploratory action. **Highest noise** to force attempts at unexplored behaviors. |
| $\mathbf{C}$ (Semantic Structure) | 0.10 | **Core of creativity.** Noise encourages associative thinking and hypothesis generation. |
| $\mathbf{E_{env}}$ Output | 0.05 | Attention noise. **Lowest noise** to suppress perceptual randomness and reduce $\mathbf{C}$'s burden. |
| Non-Applied Layers | 0.00 | $\mathbf{H}_{pz}, \mathbf{U}_{pz}, \mathbf{E_{self}}, \mathbf{E_{ctrl}}$ are completely excluded from noise to maintain internal stability and coherence. |
| Decay Rate ($\lambda_{\text{anneal}}$) | $1 \times 10^{-6}$ | Very slow decay to maintain long-term exploration opportunities even after skill stabilization. |
| Stop Threshold ($\sigma^2_{\min}$) | 0.01 | Prevents falling into local optima by maintaining a small, persistent flicker of randomness. |

---

<a id="initial-values"></a>
## V. ⚛️ Integration of Final Initial Values and Hyperparameters

| Category | Parameter | Defined Value | Role |
| :--- | :--- | :--- | :--- |
| Intelligence Core Dim. | $\mathbf{n_C}$ (Semantic Structure) | 512 | Maintains smooth information flow, starting with zero knowledge ($\mathbf{C}(0) = \mathbf{0}$). |
| Self-Model Dim. | $\mathbf{n_S}$ ($\mathbf{E_{self}}$) | 128 | Initial self-perception starts from a diverse, random state ($\sim \mathcal{U}(-1, 1)$). |
| Learning Stabilization | $\mathbf{T_{\text{BPTT}}}$ (Time Window) | 16 | Defines the scope of immediate responsibility. |
| | $\eta_X$ (Learning Rate) | $1 \times 10^{-4}$ | Promotes cautious evolution of intelligence. |
| | $\text{Clip Norm}$ (Max Gradient Norm) | 5.0 | Prevents gradient explosion and stabilizes learning. |
| Emotional Initial State | $\mathbf{H}_{pz}(0), \mathbf{U}_{pz}(0)$ | $\mathbf{0.0}$ | Happiness and anxiety start from neutral. |
| | $H_{\text{base}}, U_{\text{base}}$ | $\mathbf{0.0}$ | Initial set-points are neutral, only raised/stabilized through evolution. |
