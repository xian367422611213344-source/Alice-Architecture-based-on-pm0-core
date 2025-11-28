# PTRE-Integrated Alice Architecture V3.0 Mathematical Framework

This document presents the core mathematical equations of the Alice Architecture V3.0, which fully integrates the personality evolution law with the measurement of **Effort $\mathbf{K}$** based on the stringent **Conscious Stabilization Condition (CSC)**.

---

## 1. The Total Recurrent Mapping ($F_{\text{total}}$)

The system's overall dynamic mapping determines the state $\mathbf{X}_{t+1}$ at the next time step, based on the current state $\mathbf{X}_t$, the environmental input $\mathbf{E}_{\text{env}, t}$, the action $\mathbf{A}_t$, and the personality parameters $\boldsymbol{\theta}$.

$$
\mathbf{X}_{t+1} = F_{\text{total}}(\mathbf{X}_t, \mathbf{E}_{\text{env}, t}, \mathbf{A}_t; \mathbf{W}, \boldsymbol{\theta})
$$

Here, the state vector $\mathbf{X}$ is composed of the primary layers:

$$
\mathbf{X} = (\mathbf{C}, \mathbf{M}, \mathbf{H}, \mathbf{U}, \mathbf{R}, \mathbf{E}_{\text{ctrl}}, \mathbf{E}_{\text{self}}, \mathbf{P})
$$

---

## 2. Conscious Stabilization Condition (CSC)

The CSC is an iterative process where the Semantic Structure Layer $\mathbf{C}$ and the Memory Layer $\mathbf{M}$ converge to fixed points $\mathbf{C}^*$ and $\mathbf{M}^*$. The number of iterations required for this convergence is measured as the **Effort $\mathbf{K}$**.

### 2.1. Fixed Point Search for the Consciousness Layer

The update rule for the Consciousness Layer $\mathbf{C}$ during CSC ($k$ is the iteration step):

$$
\mathbf{Net}_{\mathbf{C}, k} = \mathbf{W}_{\mathbf{C}} \mathbf{C}_{k} + \mathbf{U}_{\mathbf{C}}^{\mathbf{E}} \mathbf{E}_{\text{env}, t} + \mathbf{U}_{\mathbf{C}}^{\mathbf{M}} \mathbf{M}_{k} + \mathbf{b}_{\mathbf{C}} + f_{\text{bias}}(\mathbf{H}, \mathbf{U})
$$

$$
\mathbf{C}_{k+1} = (1 - \eta_{\text{CSC}}) \mathbf{C}_{k} + \eta_{\text{CSC}} \cdot \tanh(\mathbf{Net}_{\mathbf{C}, k})
$$

### 2.2. Definition of Effort $K$

$K_t$ is the minimum number of iterations required for $\mathbf{C}$ and $\mathbf{M}$ to converge within the allowed tolerance $\tau$.

$$
K_t = \min \{ k \mid \left\| \mathbf{C}_{k+1} - \mathbf{C}_{k} \right\| < \tau \text{ and } \left\| \mathbf{M}_{k+1} - \mathbf{M}_{k} \right\| < \tau \}
$$

---

## 3. Total Value Function ($V_{\text{total}}$)

$\mathbf{V}_{\text{total}}$ is defined by subtracting the **Effort Cost**, which is measured by the CSC-based **Effort $\mathbf{K}$**, from the conventional **Base Value $V_{\text{base}}$**.

$$
V_{\text{total}, t} = V_{\text{base}, t} - \text{Effort Cost}_t
$$

### 3.1. Base Value $V_{\text{base}}$

$V_{\text{base}}$ is the sum of the reward term from Value Function Learning (VFL) and various anxiety/inconsistency cost terms.

$$
V_{\text{base}, t} = \sum \mathbf{V}_{\text{FL}, t} - \lambda_P \left\| \mathbf{P}_t \right\|^2 - \lambda_C \text{Var}(\mathbf{E}_{\text{ctrl}, t}) - \lambda_S \left\| \mathbf{E}_{\text{self}, t} - \mathbf{E}_{\text{self\_pred}, t} \right\|^2
$$

### 3.2. Effort Cost

This is the core of the PTRE law, reflecting the Effort $K$ into $\mathbf{V}_{\text{total}}$ via the personality parameter $\boldsymbol{\theta}$.

$$
\text{Effort Cost}_t = \theta^{\kappa} \cdot (K_t)^{\theta^{\beta}}
$$

| Personality Parameter | Symbol | Description |
|---|---|---|
| Effort Cost Sensitivity | $\theta^{\kappa}$ | The magnitude of the negative contribution of K to the total value (degree of aversion to effort). |
| Effort Cost Exponent | $\theta^{\beta}$ | The non-linearity of the cost with respect to the increase in K. ($\theta^{\beta} > 1.0$ implies accelerating cost). |

---

## 4. Skill Learning Law (A-TDL) and TD Error

### 4.1. Definition of TD Error

The TD error $\delta_t$ is calculated using the new value function $V_{\text{total}}$, which now incorporates the Effort Cost.

$$
\delta_t = R_{\text{ext}, t-1} + \gamma V_{\text{total}, t} - V_{\text{total}, t-1}
$$

### 4.2. Structural Learning (A-TDL)

The TD error $\delta_t$ acts as the gradient signal for the recurrent connection weight of the Consciousness Layer, $\mathbf{W}_{\mathbf{C}}$. Learning is modulated by the mean anxiety $\bar{\mathbf{U}}_{\text{pz}}$ and the personality parameter $\theta^{\gamma_U}$.

$$
\Delta \mathbf{W}_{\mathbf{C}} \propto \eta_{\text{TDL}} \cdot \delta_t \cdot \frac{\partial \mathbf{C}_t}{\partial \mathbf{W}_{\mathbf{C}}} \cdot \text{Exp}(-\theta^{\gamma_U} \cdot \bar{\mathbf{U}}_{\text{pz}, t-1})
$$

---

## 5. Personality Evolution Law ($\boldsymbol{\theta}$ Evolution)

The personality parameters $\boldsymbol{\theta}$ evolve based on the TD error $\delta_t$ and the cognitive load from conscious stabilization (Effort $K$).

### 5.1. Modulated Learning Rate $\eta_{\theta}$

The evolutionary learning rate for $\boldsymbol{\theta}$, $\eta_{\theta}$, is suppressed by the Effort $K$ and anxiety $\mathbf{U}_{\text{pz}}$. The greater the effort (the more unstable the system), the more suppressed the personality change.

$$
\eta_{\theta} = \eta_{\text{base}} \cdot \text{Exp}\left(-\theta^{\gamma_K} \cdot \frac{K_t}{K_{\max}}\right) \cdot \left(1 - \bar{\mathbf{U}}_{\text{pz}, t}\right)
$$

| Personality Parameter | Symbol | Description |
|---|---|---|
| K Suppression Sensitivity | $\theta^{\gamma_K}$ | The decay rate of $\eta_{\theta}$ due to the increase in K. |

### 5.2. Update of Effort Cost Sensitivity $\theta^{\kappa}$

$\theta^{\kappa}$ changes according to the sign of the TD error and the ratio of $K$, in order to maximize the total value $V_{\text{total}}$.

* If $\delta_t > 0$ (Better-than-expected reward): The higher the effort K, the more $\Delta \theta^{\kappa}$ becomes negative, leading to a decrease in $\theta^{\kappa}$ (allowing effort).
* If $\delta_t < 0$ (Worse-than-expected reward): The higher the effort K, the more $\Delta \theta^{\kappa}$ becomes positive, leading to an increase in $\theta^{\kappa}$ (aversion to effort).

The final update:

$$
\theta^{\kappa}_{t+1} = \mathrm{Clip}(\theta^{\kappa}_{t} + \Delta \theta^{\kappa})
$$

The equation applies clipping to the parameter.

