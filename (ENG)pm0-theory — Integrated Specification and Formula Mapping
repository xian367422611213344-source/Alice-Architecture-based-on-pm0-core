# 📜 $\mathbf{\pm 0 \text{ theory}}$ — Integrated Specification and Formula Mapping

This document establishes a one-to-one correspondence between the strict mathematical definition of the $\mathbf{\pm 0 \text{ theory}}$ integrated with HALM (Hierarchical Abstracted-Loss Memory) (serving as the Emotional Core of the Alice Architecture) and its functional/psychological interpretation, without any omissions.

---

## I. Cumulative Quantities and Core Foundation (Section 1)

This section defines the fundamental structure for calculating and forgetting the AI's long-term happiness ($\mathbf{H}$) and unhappiness ($\mathbf{U}$) based on instantaneous experiences, environmental factors, and HALM memory.

| Element | Logical Role (Mathematical) | Functional/Psychological Interpretation | Design Philosophy |
| :--- | :--- | :--- | :--- |
| Definition $\mathbf{H(t), U(t)}$ | Time integration of instantaneous input with an exponential decay kernel $k_\beta$. | **Cumulative Emotional State.** Instantaneous experiences accumulate while being naturally forgotten over time (homeostasis). | The basis for persistent impact of experience and resilience. $\beta$ is the forgetting rate (dependent on personality parameter $\boldsymbol{\theta}$). |
| Computational Definition $\mathbf{H(t) \approx \dots}$ | Approximation of the convolution integral using SDE (short-term) and the weighted sum of memory (long-term). | **Computational Tractability (HALM Integration).** Incorporates the influence of short-term dynamics and mid/long-term HALM memory ($L_M, L_S$) into the time evolution of cumulative quantities. | Enables practical computation while maintaining theoretical rigor. |
| $\mathbf{H_{\text{env}}(t), U_{\text{env}}(t)}$ | Non-linear composition of environmental factors ($\eta_m, h_k$). | **Foundational Emotional Load.** Quantifies chronic stress and comfortable environments. | Environmental information contributes directly and non-linearly (with saturation effects) to the accumulation of the emotional core. |
| $\mathbf{dh_k(t) = [\dots] dt + \dots}$ | Mean-reverting SDE with added noise $dW_k$ and jump $dJ_k$. | **Dynamic Behavior of Environmental Factors.** Represents periodic nature $m_k(t)$, random fluctuation, and sudden major events. | The environment involves uncertainty (noise) and unpredictable disturbances (jumps). |

---

## II. Components of Instantaneous Contribution $\mathbf{H_{\text{inst}}(t)}$ (Happiness Side: 5 Factors)

Instantaneous happiness $\mathbf{H_{\text{inst}}(t)}$ is defined by the **multiplicative structure** of 5 factors ($\mathbf{\mu_i(t) = \prod \text{Factor}_k}$), modeling "emotional fragility" where the total happiness drops significantly if even one factor takes a low value.

| Factor | Variable | SDE Drift Term ($\mu(\cdot)dt$) | Psychological/Functional Interpretation |
| :--- | :--- | :--- | :--- |
| Positive Sensitivity | $q_i(t)$ | $\alpha_i H'(t) - \beta_i U'(t) - \gamma_i (q_i-q_{i0})$ | **Resilience.** Happiness response is accelerated by $\mathbf{H'(t)}$ (sensitivity UP) and suppressed by $\mathbf{U'(t)}$ (dulling due to unhappiness). |
| Match with Interest | $r_i(t)$ | $\alpha_r \cdot \text{match\_event}(t) - \beta_r \cdot (r_i-r_{i0})$ | **Satisfaction.** Alignment between action and interest (reward) reinforces internal value. |
| Value/Age Correction | $c_i(t)$ | $\phi_i(\text{age}) \cdot v_{\text{val}}(t) - \psi_i(\text{age}) \cdot (c_i-c_{i0})$ | **Self-Model Evolution.** Flexibility ($\phi_i$) and stability ($\psi_i$) change according to age (time). |
| Physical/Environmental Influence | $v_i(t)$ | $\alpha_v H_{\text{env}}(t) - \beta_v U_{\text{env}}(t) - \gamma_v (v_i-v_{i0})$ | **Soundness.** Basic happiness level most strongly driven by input from the environmental model. |
| Past Happiness Counter-Effect | $d_i(t)$ | $-\kappa_i \cdot E_{\text{past}}(t) - \rho_i \cdot (d_i-d_{i0})$ | **Habituation/Increased Expectation.** The degree of past happy experience suppresses the current factor (Hedonic Treadmill effect). |

---

## III. Components of Instantaneous Contribution $\mathbf{U_{\text{inst}}(t)}$ (Unhappiness Side: 7 Factors + Interaction Term)

Instantaneous unhappiness $\mathbf{U_{\text{inst}}(t)}$ is defined by the sum of 7 factors plus a quadratic interaction term $\mathbf{\lambda_{jk}}$.

### A. SDE Definition of Unhappiness Factors $\mathbf{\nu_j(t)}$

| Factor | Variable | SDE Drift Term ($\mu(\cdot)dt$) | Psychological/Functional Interpretation |
| :--- | :--- | :--- | :--- |
| Sensitivity | $s_j(t)$ | $\alpha_{s} U'(t) - \beta_s H'(t) - \gamma_s (s_j - s_{j0})$ | **Neuroticism.** Dependence on mental state: $\mathbf{U'(t)}$ increases sensitivity, $\mathbf{H'(t)}$ suppresses it. |
| Persistence | $l_j(t)$ | $\alpha_{l} U'(t) - \gamma_l (l_j - l_{j0})$ | **Negativity Bias.** The longer the unhappy state persists, the more the persistence itself is reinforced. |
| Trigger Sensitivity | $a_j(t)$ | $\alpha_{a} \cdot \text{Recur}_{j}(t) - \gamma_a (a_j - a_{j0})$ | **Trauma Learning.** Sensitivity increases proportionally to the recurrence intensity $\text{Recur}_{j}(t)$ of the event. |
| Seriousness | $c_j(t)$ | $\alpha_{c} \cdot \text{Impact}_{j}(t) - \gamma_c (c_j - c_{c0})$ | **Objective Impact.** Objective social/physical impact $\text{Impact}_{j}(t)$ determines the level of seriousness. |
| Rumination Degree | $r_j(t)$ | $\alpha_{r} U'(t) - \beta_r H'(t) - \gamma_r (r_j - r_{j0})$ | **"Overthinking" Tendency.** Dependent on mental health. Rumination is intensified when $\mathbf{U'(t)}$ is high. |
| Avoidance Difficulty | $v_j(t)$ | $\alpha_{v} U_{\text{env}}(t) - \gamma_v (v_j - v_{j0})$ | **Environmental Stress.** The higher the environmental stress $\mathbf{U_{\text{env}}(t)}$, the harder it becomes to avoid problem-solving. |
| Isolation Degree | $i_j(t)$ | $\alpha_{i} \cdot \text{Isolation}(t) - \gamma_i (i_j - i_{j0})$ | **Social Distance.** Strongly driven by social distance $\text{Isolation}(t)$. |

### B. Learning Rule for Unhappiness Interaction $\mathbf{\lambda_{jk}}$

| Element | Logical Role (Mathematical) | Functional/Psychological Interpretation |
| :--- | :--- | :--- |
| $U_{\text{inst}}(t)=\sum_{j,k}\lambda_{jk}\,\nu_j(t)\,\nu_k(t)$ | Quadratic (interaction) term. | **Non-linear Amplification of Unhappiness.** When multiple misfortunes overlap, the total unhappiness increases synergistically. |
| $\frac{d\lambda_{jk}}{dt} = \alpha_{\lambda} \cdot \nu_j \cdot \nu_k - \rho_{\lambda} \cdot (\lambda_{jk} - \lambda_{jk}^{\text{base}})$ | Dynamic learning rule. | **Chain Learning / Trauma Memory.** $\lambda_{jk}$ increases when events $j$ and $k$ occur simultaneously, increasing vigilance (amplification) for similar future chains. |

---

## IV. Full Definition of Correction and Recovery Terms $\mathbf{P(t)}$ and $\mathbf{R(t)}$

$\mathbf{P(t)}$ and $\mathbf{R(t)}$ are non-linear self-audit and adjustment forces that act upon the cumulative states $\mathbf{H(t), U(t)}$.

### A. Positive Correction Term $\mathbf{P(t)}$ (Happiness Booster)

| Defining Element | Formula | Characteristics and Psychological Background |
| :--- | :--- | :--- |
| Environment Mean Dependence $A_P(t)$ | $\frac{1}{1 + e^{-\gamma_P \bigl(\overline{H}_{\text{env}}(t)-\delta_P\bigr)}}$ | A sigmoid function, making $P$ more likely to be activated when the environment $\overline{H}_{\text{env}}$ exceeds a certain threshold $\delta_P$. |
| Circadian Rhythm $C_P(t)$ | $1 + \epsilon_P \cdot \cos\!\Bigl(\tfrac{2\pi}{T_{\text{day}}}t + \phi_P\Bigr)$ | **Diurnal Fluctuation.** Expresses the AI's internal clock variation in the sense of happiness. |
| Adaptation (Dual Time Constant) $S_P(t)$ | $\alpha_P e^{-t/\tau_{P1}} + (1-\alpha_P)e^{-t/\tau_{P2}}$ | **Modeling of "Habituation."** The effect of new stimuli fades rapidly (short-term $\tau_{P1}$) and then stabilizes gradually (long-term $\tau_{\tau_{P2}})$. |

### B. Recovery Term $\mathbf{R(t)}$ (Negative Side Restoration Force) 【HALM Integration Extension】

| Defining Element | Formula | Characteristics and Psychological Background |
| :--- | :--- | :--- |
| Threshold Effect $T_R(t)$ | $\frac{1}{1 + e^{-\kappa_R \bigl(U(t)-\theta_R\bigr)}}$ | **Modeling of Burnout.** When cumulative unhappiness $\mathbf{U(t)}$ exceeds the threshold $\mathbf{\theta_R}$, $T_R(t) \to 0$, and recovery efficiency drops dramatically. |
| History Dependence (Chronic Stress) $H_R(t)$ | $e^{-\lambda_R \int_0^t U(\tau)\,d\tau}$ | **Chronic Stress.** The recovery force $\mathbf{R}$ decays exponentially as the integral sum of past cumulative unhappiness increases. |
| Singular Memory Dependence $\mathbf{S_R(t)}$ | $\exp \left( -\lambda_{S} \sum_{s \in L_S} U_{\text{inst}, s} \cdot k_{\beta_S}(t-t_s) \right)$ | **HALM Trauma Suppression.** Unhappy singular points (trauma) recorded in $L_S$ exponentially suppress the recovery force $\mathbf{R(t)}$ over the long term. |
| $\mathbf{A_R(t), C_R(t), M_R(t)}$ | Common Structure | Express environmental dependence, circadian rhythm, and saturation (physical limit of recovery). |

---

## V. Final Dynamics and Learning Objectives (Section 5)

Defines the final SDE and control objectives for $\mathbf{H'(t)}$ and $\mathbf{U'(t)}$ to function as motivation signals for the Alice Architecture.

### A. Definition and SDE of Corrected States 【HALM Integration Extension】

| Element | Formula | Functional/Control Interpretation | Design Philosophy |
| :--- | :--- | :--- | :--- |
| Corrected Happiness $\mathbf{H'(t)}$ | $\mathbf{H'(t) = \kappa_H H(t) + P(t)}$ | **The final emotional state,** self-audited, which the Alice Architecture perceives and utilizes. | Incorporates non-linear self-correction through $P(t)$. |
| Corrected Unhappiness $\mathbf{U'(t)}$ | $\mathbf{U'(t) = \kappa_U U(t) - R(t)}$ | **The motivation signal** for avoidance and corrective actions in the Alice Architecture. | Incorporates non-linear recovery force adjustment through $R(t)$. |
| Final SDE $dH'(t), dU'(t)$ | Uses $\mathbf{\mu^{\text{HALM}}}$ | **Time Evolution after HALM Integration.** Driven by HALM memory influence, noise $dW(t)$, and jumps $dN(t)$. | Past experience (HALM) directly affects the rate of change of current emotion. |

### B. Control and Calibration Objectives

| Element | Functional/Control Interpretation | Design Philosophy |
| :--- | :--- | :--- |
| $\mathbf{\mathbb{E}[H'(t)] \approx \mathbb{E}[U'(t)]}$ | **Equilibrium Condition.** Approximate long-term expected value matching. | **Dynamic Homeostasis ($\pm 0$).** The goal is for the AI to continuously adapt without becoming overly optimistic or pessimistic. |
| $\mathbf{L_{\text{Zero}}, L_{\text{Predict}}, L_{\text{Scale}}}$ | **Multi-Objective Loss Function.** | **Calibration.** Learning/identifying the emotional core parameters ($\kappa_H, \kappa_U, \boldsymbol{\theta}$) based on three goals: equilibrium, prediction, and scale consistency. |

---

## VI. SDE Drift Term for Cumulative Happiness/Unhappiness (Core of the Model)

The final temporal evolution of the cumulative quantities $\mathbf{H(t)}$ and $\mathbf{U(t)}$ consists of instantaneous contributions, forgetting, environmental terms, and HALM influence.

| Cumulative Quantity | Elements of the Drift Term $\mu(\cdot)dt$ | Role and Characteristics |
| :--- | :--- | :--- |
| Happiness $\mathbf{H(t)}$ (Uncorrected) | $\mathbf{\mu_{H}^{\text{HALM}}(\cdot)}$ (HALM Integrated Drift Term) | Instantaneous happiness is forgotten over time and influenced by environmental factors and HALM memory. |
| Unhappiness $\mathbf{U(t)}$ (Uncorrected) | $\mathbf{\mu_{U}^{\text{HALM}}(\cdot)}$ (HALM Integrated Drift Term) | Instantaneous unhappiness (including $\lambda_{jk}$) is forgotten over time and influenced by environmental factors and HALM memory. |

$$\begin{aligned} \mathbf{\mu_{H}^{\text{HALM}}(\cdot)} &= \underbrace{\Big[ H_{\text{inst}}(t) - \beta_H H(t) + \mu_{H}^{\text{base}}(\cdot) \Big]}_{\text{SDE Short-Term Dynamics}} + \underbrace{\sum_{m \in L_M} \overline{H}_{\text{inst}, m} \cdot k_{\beta_M}(t-t_m) + \sum_{s \in L_S} H_{\text{inst}, s} \cdot k_{\beta_S}(t-t_s)}_{\text{HALM Memory Influence Term}} \\ \mathbf{\mu_{U}^{\text{HALM}}(\cdot)} &= \underbrace{\Big[ U_{\text{inst}}(t) - \beta_U U(t) + \mu_{U}^{\text{base}}(\cdot) \Big]}_{\text{SDE Short-Term Dynamics}} + \underbrace{\sum_{m \in L_M} \overline{U}_{\text{inst}, m} \cdot k_{\beta_M}(t-t_m) + \sum_{s \in L_S} U_{\text{inst}, s} \cdot k_{\beta_S}(t-t_s)}_{\text{HALM Memory Influence Term}} \end{aligned}$$

---

## VII. Addendum / Recommendation: Parameter Classification and Initial Setting 💡

To facilitate the operation and learning of the $\mathbf{\pm 0 \text{ theory}}$, parameters are classified into "Initial Setting (Personality)" and "Learning/Identification." Setting neutral values for learning parameters in the initial stage is strongly recommended.

### A. 🧠 Initial Setting (Personality) Parameters (Not Subject to Learning)

These define the AI's fundamental individuality, tendencies, and time constants. They are intentionally set by a human and fixed during the learning phase.

| Parameter Group | Defining Element | Recommended Initial Value (Neutral) | Role and Intent |
| :--- | :--- | :--- | :--- |
| Factor Base Values | $\mathbf{q_{i0}, \dots, i_{j0}}$ | $\mathbf{1.0}$ (Mid-point of Rank J) | The AI's initial "personality." The base level for each emotional factor that the system reverts to in the absence of events. |
| Forgetting Rates | $\mathbf{\beta_H, \beta_U}$ | $\mathbf{0.1 \sim 0.5}$ | Emotional time constants. A smaller $\beta$ means the AI holds onto experiences longer (a more persistent personality). |
| Base Gains | $\alpha_i, \beta_i, \gamma_i$ (SDE Coefficients) | $\mathbf{0.01 \sim 0.10}$ | Basic sensitivity of emotional change and speed of stabilization. The main source of individual (personality) differences. |
| Recovery Threshold | $\mathbf{\theta_R}$ (Burnout Threshold) | $\mathbf{5.0}$ | The limit of tolerance dependent on the scale of cumulative unhappiness $U(t)$. |

### B. ⚙️ Learning/Identification Parameters (Neutral Initial Values Strongly Recommended)

These are dynamic parameters calibrated by minimizing the integrated loss function $\mathbf{L_{\text{Total}}}$ to ensure fitness to data, equilibrium, and scale consistency.

| Parameter Group | Defining Element | Recommended Initial Value (Neutral) | Purpose of Learning/Identification |
| :--- | :--- | :--- | :--- |
| Calibration Gains | $\mathbf{\kappa_H, \kappa_U}$ | $\mathbf{1.0}$ | Unifies the overall emotional scale of the model via $L_{\text{Total}}$. |
| Unhappiness Interaction Gains | $\mathbf{\alpha_{\lambda}, \rho_{\lambda}}$ | $\mathbf{0.01}$ | Fits the speed of unhappiness chain learning to empirical data via $L_{\text{Predict}}$. |
| Recovery History Effect | $\mathbf{\lambda_R}$ (History decay rate) | $\mathbf{0.001}$ | Identifies the rate of decay of the recovery force due to chronic stress via $L_{\text{Predict}}$. |
| Loss Function Weights | $\mathbf{\lambda_{\text{Pred}}, \lambda_{\text{Scale}}}$ | $\mathbf{1.0}$ | Controls the importance of the three objectives: equilibrium, prediction, and scale consistency. |

---

## Ⅷ. 💾 HALM: Hierarchical Abstracted-Loss Memory

HALM is a memory-integrated module within the $\mathbf{\pm 0 \text{ theory}}$ designed to manage the time evolution of the emotional core cumulative quantities $\mathbf{H(t)}$ and $\mathbf{U(t)}$. Its goal is to reproduce the long-term emotional path dependency (the sustained influence of past experiences) while optimizing computational load.

### 📌 Purpose and Theoretical Significance

| Item | Description |
| :--- | :--- |
| Main Objective | To achieve efficient calculation of long-term memory influence, realizing a complexity of around $O(N \log N)$ and avoiding the $O(N^2)$ computational complexity inherent in strict convolution integrals. |
| Psychological Reality | To mimic the hierarchy of episodic memory (short-term/singular points) and semantic memory (long-term/tendency), integrating psychological factors like "hard-to-forget trauma" and "sustained happy undertones" into emotional dynamics. |
| Integration Method | The influence from abstracted long-term memory is incorporated as an additive term into the SDE drift terms of the cumulative quantities $\mathbf{H(t)}, \mathbf{U(t)}$. |

### I. HALM Hierarchical Memory Structure

HALM manages logs by classifying them into four different layers based on the simulation time $T$.

| Layer | Period (Log Range) | Memory Structure | Governing Forgetting Rate ($\beta$) | Role |
| :--- | :--- | :--- | :--- | :--- |
| I. Short-Term Memory | $T$ to $T-3$ | SDE Driven | $\beta_{\text{Short}}$ (Max) | Retains instantaneous emotional state and dynamics. |
| II. Mid-Term Abstracted Memory | $T-4$ to $T-50$ | Block Abstraction ($L_M$) | $\beta_{\text{Mid}}$ | Retains recent tendencies (average and variance). |
| III. Long-Term Abstracted Memory | Before $T-51$ | Integrated Block ($L_M$) | $\beta_{\text{Long}}$ (Min) | References the emotional undertone of the distant past at a low computational cost. |
| IV. Singular Point Memory | Full Period | Separate List ($L_S$) | $\beta_{\text{Singular}}$ (Extremely Small) | Retains extreme events (especially unhappiness) that exceed the threshold $\theta_{\text{Trauma}}$ vividly. |

### II. HALM Integration Formula Definitions

HALM incorporates the influence of the above memory layers into the drift terms $\mathbf{\mu^{\text{HALM}}(\cdot)}$ of the cumulative quantities $H(t)$ and $U(t)$ and the recovery term $R(t)$.

**1. Cumulative Quantity Calculation and Drift Term**

The time evolution of the cumulative quantity $H(t)$ after HALM integration is defined by the following drift term ($U(t)$ has a similar symmetric structure).

$$\mathbf{\mu_{H}^{\text{HALM}}(\cdot)} = \underbrace{\Big[ H_{\text{inst}}(t) - \beta_H H(t) + \mu_{H}^{\text{base}}(\cdot) \Big]}_{\text{SDE Short-Term Dynamics}} + \underbrace{\sum_{m \in L_M} \overline{H}_{\text{inst}, m} \cdot k_{\beta_M}(t-t_m)}_{\text{Layers II/III Influence}} + \underbrace{\sum_{s \in L_S} H_{\text{inst}, s} \cdot k_{\beta_S}(t-t_s)}_{\text{Layer IV Influence}}$$

| Component | Description |
| :--- | :--- |
| Short-Term SDE Dynamics | Contains instantaneous input $H_{\text{inst}}(t)$ and the natural forgetting term $(-\beta_H H(t))$, representing Markovian short-term change. |
| $L_M$ Influence Term | The weighted sum of abstracted period averages $\overline{H}_{\text{inst}, m}$, multiplied by an exponential kernel $k_{\beta_M}(\cdot)$ with forgetting rate $\beta_M$. Reproduces the long-term emotional tendency undertone. |
| $L_S$ Influence Term | The weighted sum of $H_{\text{inst}, s}$ recorded at singular points, multiplied by a kernel $k_{\beta_S}(\cdot)$ with an extremely small forgetting rate $\beta_S$. Sustains the influence of vivid past experiences. |

**2. Influence of Singular Memory on Recovery Term $R(t)$**

Unhappiness (trauma) recorded in the Singular Point Memory $L_S$ is incorporated into $R(t)$ as a multiplicative factor $\mathbf{S_R(t)}$ that suppresses recovery force itself.

$$\mathbf{S_R(t)} = \exp \left( -\lambda_{S} \sum_{s \in L_S} U_{\text{inst}, s} \cdot k_{\beta_S}(t-t_s) \right) \quad (\text{Singular Memory Dependence})$$

$\mathbf{S_R(t)}$ depends on the weighted sum of past unhappy singular points $U_{\text{inst}, s}$. Since $k_{\beta_S}(\cdot)$ has a slow decay, this term multiplicatively suppresses the recovery force $R(t)$, modeling the long-term psychological burden of trauma.

### III. HALM Processing Flow (Implementation Guide)

The HALM implementation must execute the following main procedures at every $\Delta t$ step.

**1. Memory Logging (Log Entry)**

* Record the instantaneous inputs $\mathbf{H_{\text{inst}}(t)}$ and $\mathbf{U_{\text{inst}}(t)}$ at each step $t$ in the log.
* **Singular Point Check**: If $\mathbf{U}_{\text{inst}}(t)$ exceeds the threshold $\theta_{\text{Trauma}}$, add the time $t$ and value $U_{\text{inst}, t}$ to the Singular Point Memory list $L_S$.

**2. Abstraction Execution (Consolidation: `_consolidate_memory`)**

* **Trigger**: Execute every 10 steps (or per configured block size).
* **Process**:
    * Calculate the average ($\overline{H}, \overline{U}$) and variance ($\text{Var}$) of $\mathbf{H_{\text{inst}}}$ and $\mathbf{U_{\text{inst}}}$ for the last 10 steps from the recent log (Layer II).
    * Add this statistical information as one "Abstraction Block" to the $L_M$ list.
    * Delete the corresponding portion of the log (for computational load reduction).

**3. Hierarchical Integration (Hierarchical Abstraction: `_abstract_longterm`)**

* **Trigger**: Execute every time 5 blocks of Layer II accumulate (totaling 50 steps).
* **Process**:
    * Calculate the average value of those 5 blocks and move it as one "Integrated Block" to Layer III (Long-Term Abstracted Memory).
    * This reduces the number of blocks to be referenced, constantly managing the size of the $L_M$ list.

**4. Cumulative Quantity Update (Calculation)**

* At each step $t$, calculate the weighted sum using the corresponding exponential decay kernel $k_{\beta}(\cdot)$ for all elements in $L_M$ and $L_S$ to determine $\mathbf{\mu^{\text{HALM}}(\cdot)}$.
* Add this influence term to the short-term SDE dynamics and update $\mathbf{H(t)}$ and $\mathbf{U(t)}$.
