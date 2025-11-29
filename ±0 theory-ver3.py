import numpy as np
from typing import Dict, Any, List, Optional
import math

# --- 0. 定数と次元設定 ---

# 変更点 1: 創造性因子 (q3) を追加するため、幸福因子次元数を 2 -> 3 に変更
NUM_H_FACTORS = 3  # 幸福因子 (H_inst, eta_m, 創造性 q3) の次元数
NUM_U_FACTORS = 3  # 不幸因子 (U_inst, h_k) の次元数


# --- 1. 定義済みカーネルと補助関数 (共通) ---


def exponential_decay_kernel(t: float, tau: np.ndarray, beta: float) -> np.ndarray:
    """指数減衰カーネル k_beta(t - tau) を計算する。"""
    if tau.size == 0:
        return np.array([])
    # tau <= t の要素のみを考慮 (過去のイベントのみ)
    indicator_func = (tau <= t).astype(float)
    decay_term = np.exp(-beta * (t - tau))
    return decay_term * indicator_func


def phi_age(age: float, params: Dict) -> float:
    """柔軟性関数 $\phi_i(t)$ (Exponential Decay)"""
    return params['phi_max'] * math.exp(-params['lambda_phi'] * age)


def psi_age(age: float, params: Dict) -> float:
    """安定性関数 $\psi_i(t)$ (Logistic Function)"""
    p = params
    logistic_term = 1.0 / (1.0 + math.exp(-p['k_psi'] * (age - p['age_1/2'])))
    return p['psi_min'] + (p['psi_max'] - p['psi_min']) * logistic_term


class AliceEmotionalCore:
    def __init__(self, dt: float = 0.1, rng_seed: Optional[int] = None):
        # --- 2. 時間と確率過程 ---
        self.dt = dt  # 時間ステップ (0.1 day)
        self.t = 0.0  # 現在のシミュレーション時間
        self.rng = np.random.default_rng(rng_seed)

        # --- 3. パラメータ群 (全フェーズで確定した厳密なパラメータを含む) ---
        self.params = self._initialize_parameters()

        # --- 4. 状態変数 ---
        self.state = self._initialize_state()

        # --- 5. HALM ログとリスト ---
        self.log_H_inst: List[float] = []
        self.log_U_inst: List[float] = []
        self.log_time: List[float] = []

        # L_M: 中期/長期抽象記憶 (レイヤー II/III)
        self.L_M: List[Dict[str, Any]] = []
        # L_S: 特異点記憶 (レイヤー IV)
        self.L_S: List[Dict[str, Any]] = []

        # HALM/SDE 補助ロギング
        self.log_U_env_prev: float = self.params['U_env_base']
        self.log_P_prev: float = 0.0
        self.log_R_prev: float = 0.0

        # HALMカウンタ
        self.N_block = 10
        self.N_consolidation = 5
        self.log_counter = 0

    def _initialize_parameters(self) -> Dict[str, Any]:
        """全フェーズで確定したパラメータの初期設定と、創造性パラメータを定義する。"""
        P = {}
        # I. HALM/忘却率
        P.update({
            'beta_H': 0.50, 'beta': 0.50,
            'beta_Mid': 0.15, 'beta_Long': 0.01, 'beta_Singular': 0.001,
            'theta_Trauma': 3.0,
        })
        # II. SDE係数と駆動パラメータ (H因子配列サイズを NUM_H_FACTORS=3 に合わせる)
        P.update({
            # H因子 (NUM_H_FACTORS=3)
            'q_i0': np.full(NUM_H_FACTORS, 1.0), 'r_i0': np.full(NUM_H_FACTORS, 1.0),
            'c_i0': np.full(NUM_H_FACTORS, 1.0), 'v_i0': np.full(NUM_H_FACTORS, 1.0),
            'd_i0': np.full(NUM_H_FACTORS, 1.0),
            'alpha_r': 0.1, 'beta_r': 0.05,
            'kappa_i': 0.10, 'rho_i': 0.05, 'theta_i': 1.0,
            'delta_r': 0.1, 'delta_i': 0.1,
            'sigma_env': 0.1,

            # U因子 (NUM_U_FACTORS=3)
            's_j0': np.full(NUM_U_FACTORS, 1.0), 'l_j0': np.full(NUM_U_FACTORS, 1.0),
            'a_j0': np.full(NUM_U_FACTORS, 1.0), 'c_j0': np.full(NUM_U_FACTORS, 1.0),
            'r_j0': np.full(NUM_U_FACTORS, 1.0), 'v_j0': np.full(NUM_U_FACTORS, 1.0),
            'i_j0': np.full(NUM_U_FACTORS, 1.0),
            'alpha_a': 0.1, 'gamma_a': 0.05,
            'alpha_c': 0.1, 'gamma_c': 0.05,
        })
        # III. c_i(t) 成長パラメータ
        P.update({
            'phi_max': 1.0, 'lambda_phi': 0.05,
            'psi_min': 0.1, 'psi_max': 1.0, 'k_psi': 0.2, 'age_1/2': 10.0,
        })
        # IV. 環境 SDE パラメータ (H因子配列サイズを NUM_H_FACTORS=3 に合わせる)
        P.update({
            # $\eta_m$ (快適環境: N_H=3)
            'alpha_eta': np.full(NUM_H_FACTORS, 0.05), 'eta_bar': np.full(NUM_H_FACTORS, 1.0),
            'sigma_eta': np.full(NUM_H_FACTORS, 0.1), 'rho_m': np.full(NUM_H_FACTORS, 0.5), # 合成ゲイン $\rho_m$
            'lambda_J_eta': np.full(NUM_H_FACTORS, 0.001), 'b_J_eta': np.full(NUM_H_FACTORS, 0.5),

            # $h_k$ (不幸環境: N_U=3)
            'alpha_hk': np.full(NUM_U_FACTORS, 0.04), 'hk_bar': np.full(NUM_U_FACTORS, 0.5),
            'sigma_hk': np.full(NUM_U_FACTORS, 0.15), 'sigma_k_env': np.full(NUM_U_FACTORS, 0.6), # 合成ゲイン $\sigma_k$
            'lambda_J_hk': np.full(NUM_U_FACTORS, 0.005), 'b_J_hk': np.full(NUM_U_FACTORS, 1.0),

            'alpha_avg': 0.2, 'sigma_avg': 0.05,
            'U_env_base': 1.0, 'H_env_base': 1.0,
        })
        # V. 較正/学習パラメータ (既存設定を維持)
        P.update({
            'kappa_H': 1.0, 'kappa_U': 1.0,
            'lambda_Zero': 1.0, 'lambda_Predict': 1.0, 'lambda_Scale': 0.5,
            'Target_Scale_C': 1.0,
            'alpha_lambda': 0.01, 'rho_lambda': 0.01, 'lambda_base': 0.0,
            'theta_R': 5.0, 'kappa_R': 1.0, 'lambda_R': 0.001,
            'lambda_S': 0.1,
            'alpha_I': 0.05, 'gamma_I': 0.1, 'sigma_I': 0.1,
            'alpha_f': 1.0, 'beta_f': 0.5, 'alpha_s': 2.0, 'delta_s': 0.0,
        })
        # VI. 補正項 P(t), R(t) の乗法要素 (既存設定を維持)
        P.update({
            'gamma_P': 1.0, 'delta_P': 0.5, 'gamma_R': 1.0, 'delta_R': 0.5,
            'epsilon_P': 0.1, 'epsilon_R': 0.1, 'T_day': 1.0, 'phi_P': 0.0, 'phi_R': math.pi,
            'alpha_P': 0.5, 'tau_P1': 10.0, 'tau_P2': 100.0,
            'beta_P': 1.0, 'beta_R': 1.0,
        })
        # VII. 最終 SDE / 認知ジャンプ (既存設定を維持)
        P.update({
            'sigma_H_prime': 0.1, 'sigma_U_prime': 0.1,
            'lambda_H_prime': 0.05, 'b_H_prime': 1.0,
        })
        # VIII. v_val(t) SDE (既存設定を維持)
        P.update({
            'alpha_v': 0.05, 'T_trend': 365.0, 'A_v': 0.2, 'B_v': 1.0, 'sigma_v': 0.1,
            'phi_v': 0.0,
        })
        
        # 変更点 2: IX. 創造性駆動パラメータの追加
        P.update({
            # IX. 創造性駆動パラメータ (H^2駆動)
            'lambda_C': 100.0,     # \lambda_C: 創造性の二乗増幅ゲイン (超主要因)
            'kappa_C': 0.1,        # \kappa_C: 創造的学習の感度ゲイン
            'rho_C': 0.005,        # \rho_C: 創造性の忘却率
            'sigma_C': 0.01,       # \sigma_C: ランダムな閃きのノイズ
            'q3_max': 5.0,         # q_3^{max}: 創造性因子の最大飽和値
        })

        return P

    def _initialize_state(self) -> Dict[str, Any]:
        """状態変数の初期値を定義する。（NUM_H_FACTORS=3 に合わせて初期化）"""
        P = self.params
        S = {'H': 0.0, 'U': 0.0}

        # 瞬間因子 (多次元配列 $N_H=3, N_U=3$)
        S.update({'q_i': np.copy(P['q_i0']), 'r_i': np.copy(P['r_i0']), 'c_i': np.copy(P['c_i0']),
                  'v_i': np.copy(P['v_i0']), 'd_i': np.copy(P['d_i0'])})
        S.update({'s_j': np.copy(P['s_j0']), 'l_j': np.copy(P['l_j0']), 'a_j': np.copy(P['a_j0']),
                  'c_j': np.copy(P['c_j0']), 'r_j': np.copy(P['r_j0']), 'v_j': np.copy(P['v_j0']),
                  'i_j': np.copy(P['i_j0'])})

        # 環境 SDE (h_k: 不幸環境, eta_m: 快適環境)
        S.update({'h_k': np.copy(P['hk_bar']), 'eta_m': np.copy(P['eta_bar'])}) # サイズ 3 で初期化される

        # 補助 SDE / 履歴
        S.update({'v_val': P['B_v']})
        S.update({'U_hist': 0.0})
        S.update({'I': 1.0})
        S.update({'H_env_avg': P['H_env_base']})

        # $\Lambda$: 不幸相互作用ゲイン
        S.update({'lambda_sl': P['lambda_base']})
        S.update({'H_env_base': P['H_env_base'], 'U_env_base': P['U_env_base']})

        return S

    # --- 6. HALM 抽象化ロジック (変更なし) ---
    def _consolidate_memory(self):
        # ... (ロジック変更なし) ...
        if self.log_counter < self.N_block:
            return

        H_mean = np.mean(self.log_H_inst)
        U_mean = np.mean(self.log_U_inst)
        H_var = np.var(self.log_H_inst, ddof=1) if len(self.log_H_inst) > 1 else 0.0
        U_var = np.var(self.log_U_inst, ddof=1) if len(self.log_U_inst) > 1 else 0.0
        t_start = self.log_time[0]

        self.L_M.append({
            't_start': t_start, 'H_mean': H_mean, 'U_mean': U_mean,
            'H_var': H_var, 'U_var': U_var, 'size': self.N_block
        })

        self.log_H_inst.clear()
        self.log_U_inst.clear()
        self.log_time.clear()
        self.log_counter = 0

    def _abstract_longterm(self):
        # ... (ロジック変更なし) ...
        if len(self.L_M) < self.N_consolidation:
            return

        blocks = self.L_M[:self.N_consolidation]
        sizes = np.array([b['size'] for b in blocks])
        total_size = np.sum(sizes)

        H_mean_new = np.sum(np.array([b['H_mean'] for b in blocks]) * sizes) / total_size
        U_mean_new = np.sum(np.array([b['U_mean'] for b in blocks]) * sizes) / total_size
        t_start_new = blocks[0]['t_start']
        H_var_new = np.sum(np.array([b['H_var'] for b in blocks]) * sizes) / total_size
        U_var_new = np.sum(np.array([b['U_var'] for b in blocks]) * sizes) / total_size

        del self.L_M[:self.N_consolidation]
        self.L_M.append({
            't_start': t_start_new, 'H_mean': H_mean_new, 'U_mean': U_mean_new,
            'H_var': H_var_new, 'U_var': U_var_new, 'size': total_size
        })

    # --- 7. HALM 影響項の計算 (変更なし) ---
    def _calculate_halm_influence(self) -> Dict[str, float]:
        # ... (ロジック変更なし) ...
        H_influence = 0.0
        U_influence = 0.0
        # L_M 影響項
        if self.L_M:
            t_m = np.array([b['t_start'] for b in self.L_M])
            H_mean = np.array([b['H_mean'] for b in self.L_M])
            U_mean = np.array([b['U_mean'] for b in self.L_M])
            k_M = exponential_decay_kernel(self.t, t_m, self.params['beta_Long'])
            H_influence += np.sum(H_mean * k_M)
            U_influence += np.sum(U_mean * k_M)
        # L_S 影響項
        if self.L_S:
            t_s = np.array([s['t_start'] for s in self.L_S])
            H_inst_s = np.array([s['H_inst'] for s in self.L_S])
            U_inst_s = np.array([s['U_inst'] for s in self.L_S])
            k_S = exponential_decay_kernel(self.t, t_s, self.params['beta_Singular'])
            H_influence += np.sum(H_inst_s * k_S)
            U_influence += np.sum(U_inst_s * k_S)

        return {'H': H_influence, 'U': U_influence}

    # --- 8. 環境 SDE の更新 (NUM_H_FACTORS=3 に合わせるがロジック変更なし) ---

    def _update_environmental_factors(self):
        """環境因子 $\eta_m$ (3D) と $h_k$ (3D) を、平均回帰 SDE およびポアソン・ジャンプで更新する。"""
        state = self.state
        params = self.params
        dt = self.dt
        rng = self.rng

        # 乱数シードの一括生成 (サイズを N_H=3 に変更)
        Z_eta = rng.normal(size=NUM_H_FACTORS)
        Z_hk = rng.normal(size=NUM_U_FACTORS)

        # --- A. 快適環境因子 $\eta_m$ の更新 (N_H=3) ---
        eta_m = state['eta_m']
        mu_eta = params['alpha_eta'] * (params['eta_bar'] - eta_m)
        noise_eta = params['sigma_eta'] * np.sqrt(dt) * Z_eta
        jump_occur_eta = rng.random(size=NUM_H_FACTORS) < params['lambda_J_eta'] * dt
        jump_size_eta = np.where(jump_occur_eta,
                                 rng.laplace(loc=0.0, scale=params['b_J_eta']),
                                 0.0)
        state['eta_m'] += mu_eta * dt + noise_eta + jump_size_eta

        # --- B. 不幸環境因子 $h_k$ の更新 (N_U=3) ---
        h_k = state['h_k']
        mu_hk = params['alpha_hk'] * (params['hk_bar'] - h_k)
        noise_hk = params['sigma_hk'] * np.sqrt(dt) * Z_hk
        jump_occur_hk = rng.random(size=NUM_U_FACTORS) < params['lambda_J_hk'] * dt
        jump_size_hk = np.where(jump_occur_hk,
                                 rng.laplace(loc=0.0, scale=params['b_J_hk']),
                                 0.0)
        state['h_k'] += mu_hk * dt + noise_hk + jump_size_hk
        state['h_k'] = np.clip(state['h_k'], a_min=0.0, a_max=None)

    def _calculate_environmental_terms(self) -> Dict[str, float]:
        # ... (ロジック変更なし) ...
        state = self.state
        params = self.params
        # 1. 快適環境項 $H_{\text{env}}(t)$ の計算
        H_env = np.sum(params['rho_m'] * np.tanh(state['eta_m'])) # N_H=3 で計算される
        # 2. 不幸環境項 $U_{\text{env}}(t)$ の計算
        U_env = np.sum(params['sigma_k_env'] * np.tanh(state['h_k']))
        return {'H_env': H_env, 'U_env': U_env}

    # --- 9. 厳密な因子駆動項の計算 (変更なし) ---
    def _calculate_driving_terms(self, H_env: float, U_env: float) -> Dict[str, float]:
        # ... (ロジック変更なし) ...
        P = self.params
        # 1. 慣れ駆動項 $f_{\text{past}}(t)$
        L_M_influence_H = 0.0
        if self.L_M:
            t_m = np.array([b['t_start'] for b in self.L_M])
            H_mean = np.array([b['H_mean'] for b in self.L_M])
            k_M_Long = exponential_decay_kernel(self.t, t_m, P['beta_Long'])
            L_M_influence_H = np.sum(H_mean * k_M_Long)
        f_past = L_M_influence_H
        # 2. イベント駆動項 $\text{match\_event}(t)$
        match_event = 1.0 / (P['delta_r'] + abs(H_env - self.state['H']))
        # 3. 反復駆動項 $\text{Recur}_j(t)$
        recur_j = 0.0
        if self.L_M:
            t_m = np.array([b['t_start'] for b in self.L_M])
            U_var = np.array([b['U_var'] for b in self.L_M])
            k_M_Mid = exponential_decay_kernel(self.t, t_m, P['beta_Mid'])
            recur_j = np.sum(U_var * k_M_Mid)
        # 4. 衝撃駆動項 $\text{Impact}_j(t)$
        impact_j = abs(U_env - self.log_U_env_prev) / self.dt
        # 5. 孤立駆動項 $\text{Isolation}(t)$
        eta_social_avg = np.mean(self.state['eta_m'])
        isolation = 1.0 / (P['delta_i'] + eta_social_avg)

        return {
            'f_past': f_past, 'match_event': match_event, 'recur_j': recur_j,
            'impact_j': impact_j, 'isolation': isolation
        }

    # --- 10. 厳密な非線形補正・回復項の計算 (変更なし) ---
    def _calculate_correction_terms(self, H_env_avg: float) -> Dict[str, float]:
        # ... (ロジック変更なし) ...
        P = self.params
        t = self.t
        I = self.state['I']
        U = self.state['U']
        U_hist = self.state['U_hist']

        # --- P(t) の乗法要素 ---
        A_P = 1.0 / (1.0 + math.exp(-P['gamma_P'] * (H_env_avg - P['delta_P'])))
        C_P = 1.0 + P['epsilon_P'] * math.cos(2 * math.pi / P['T_day'] * t + P['phi_P'])
        S_P = P['alpha_P'] * math.exp(-t / P['tau_P1']) + (1 - P['alpha_P']) * math.exp(-t / P['tau_P2'])
        M_P = math.tanh(P['beta_P'] * I)
        P_t = A_P * C_P * S_P * M_P

        # --- R(t) の乗法要素 ---
        A_R = 1.0 / (1.0 + math.exp(-P['gamma_R'] * (H_env_avg - P['delta_R'])))
        C_R = 1.0 + P['epsilon_R'] * math.cos(2 * math.pi / P['T_day'] * t + P['phi_R'])
        T_R = 1.0 / (1.0 + math.exp(-P['kappa_R'] * (U - P['theta_R'])))
        H_R = math.exp(-P['lambda_R'] * U_hist)

        # S_R(t) (特異点記憶依存性/トラウマ抑制)
        U_inst_s = np.array([s['U_inst'] for s in self.L_S])
        if U_inst_s.size > 0:
            t_s = np.array([s['t_start'] for s in self.L_S])
            k_S = exponential_decay_kernel(t, t_s, P['beta_Singular'])
            sum_U_inst_k = np.sum(U_inst_s * k_S)
            S_R = math.exp(-P['lambda_S'] * sum_U_inst_k)
        else:
            S_R = 1.0

        M_R = math.tanh(P['beta_R'] * I)
        R_t = A_R * C_R * T_R * H_R * S_R * M_R

        return {'P': P_t, 'R': R_t}

    # --- 11. メインステップ関数 (全 SDE/ODE 完全統合) ---

    def step(self, external_inputs: Optional[Dict[str, Any]] = None):
        """シミュレーションの1ステップを実行する。（external_inputsから創造性スコアを受け取る）"""

        external_inputs = external_inputs if external_inputs is not None else {}
        
        # 1. 時刻の更新と乱数生成
        dt, t = self.dt, self.t
        P = self.params
        
        # Z_aux: v_val, H_env_avg, I, H', U', q3_SDE (サイズ 6)
        Z_aux = self.rng.standard_normal(size=6) 
        # H/U因子のSDE用 (N_H=3, N_U=3): 3*3 + 3*3 = 18 components
        Z_factors = self.rng.standard_normal(size=NUM_H_FACTORS * 3 + NUM_U_FACTORS * 3)
        
        # 2. 補助 SDE の更新: v_val(t), eta_m(t), h_k(t), H_env_avg(t)
        # v_val(t) SDE
        m_v = P['A_v'] * math.sin(2 * math.pi / P['T_trend'] * t + P['phi_v']) + P['B_v']
        dv_val_drift = -P['alpha_v'] * (self.state['v_val'] - m_v)
        self.state['v_val'] += dv_val_drift * dt + P['sigma_v'] * math.sqrt(dt) * Z_aux[0]

        # 多次元環境因子 $\mathbf{\eta}_m, \mathbf{h}_k$ の更新
        self._update_environmental_factors()

        # 環境項の非線形合成 $H_{\text{env}}, U_{\text{env}}$
        Env_terms = self._calculate_environmental_terms()
        H_env = Env_terms['H_env']
        U_env = Env_terms['U_env']

        # $\overline{H}_{\text{env}}(t)$ SDE
        dH_env_avg_drift = P['alpha_avg'] * (H_env - self.state['H_env_avg'])
        self.state['H_env_avg'] += dH_env_avg_drift * dt + P['sigma_avg'] * math.sqrt(dt) * Z_aux[1]

        # 3. 前ステップ値の準備
        H_prime_prev = P['kappa_H'] * self.state['H'] + self.log_P_prev
        U_prime_prev = P['kappa_U'] * self.state['U'] - self.log_R_prev

        # 4. 不幸相互作用項 $\lambda_{jk}$ の更新
        lambda_sl = self.state['lambda_sl']
        nu_s = np.mean(self.state['s_j']); nu_l = np.mean(self.state['l_j'])
        d_lambda_drift = P['alpha_lambda'] * nu_s * nu_l - P['rho_lambda'] * (lambda_sl - P['lambda_base'])
        self.state['lambda_sl'] += d_lambda_drift * dt
        U_inst_inter = P['lambda_sl'] * nu_s * nu_l * NUM_U_FACTORS

        # 5. 厳密な因子駆動項の計算
        Driving = self._calculate_driving_terms(H_env, U_env)
        f_past = Driving['f_past']; match_event = Driving['match_event']; recur_j = Driving['recur_j']
        impact_j = Driving['impact_j']; isolation = Driving['isolation']

        # 乱数シードのインデックス (N_H=3, N_U=3)
        idx_H_d = 0
        idx_H_r = NUM_H_FACTORS
        idx_H_c = NUM_H_FACTORS * 2
        idx_U_a = NUM_H_FACTORS * 3
        idx_U_c = NUM_H_FACTORS * 3 + NUM_U_FACTORS
        idx_U_i = NUM_H_FACTORS * 3 + NUM_U_FACTORS * 2

        # 6. 瞬間因子 SDE のベクトル更新 (d_i, r_i, c_i, a_j, c_j, i_j はサイズ 3 に対応)

        # H因子 d_i(t) の更新
        dd_i_drift = -P['kappa_i'] * f_past - P['rho_i'] * (self.state['d_i'] - P['d_i0'])
        self.state['d_i'] += dd_i_drift * dt + P['sigma_env'] * math.sqrt(dt) * Z_factors[idx_H_d:idx_H_r]

        # H因子 r_i(t) の更新
        dr_i_drift = P['alpha_r'] * match_event - P['beta_r'] * (self.state['r_i'] - P['r_i0'])
        self.state['r_i'] += dr_i_drift * dt + P['sigma_env'] * math.sqrt(dt) * Z_factors[idx_H_r:idx_H_c]

        # c_i(t) の更新
        phi = phi_age(t, P)
        psi = psi_age(t, P)
        dc_i_drift = phi * self.state['v_val'] - psi * (self.state['c_i'] - P['c_i0'])
        self.state['c_i'] += dc_i_drift * dt + P['sigma_env'] * math.sqrt(dt) * Z_factors[idx_H_c:idx_U_a]

        # U因子 a_j(t) の更新
        da_j_drift = P['alpha_a'] * recur_j - P['gamma_a'] * (self.state['a_j'] - P['a_j0'])
        self.state['a_j'] += da_j_drift * dt + P['sigma_env'] * math.sqrt(dt) * Z_factors[idx_U_a:idx_U_c]

        # U因子 c_j(t) の更新
        dc_j_drift = P['alpha_c'] * impact_j - P['gamma_c'] * (self.state['c_j'] - P['c_j0'])
        self.state['c_j'] += dc_j_drift * dt + P['sigma_env'] * math.sqrt(dt) * Z_factors[idx_U_c:idx_U_i]

        # U因子 i_j(t) の更新
        di_j_drift = 0.1 * isolation - 0.05 * (self.state['i_j'] - P['i_j0'])
        self.state['i_j'] += di_j_drift * dt + P['sigma_env'] * math.sqrt(dt) * Z_factors[idx_U_i:idx_U_i + NUM_U_FACTORS]


        # 変更点 3: 7. 創造性因子 q_i[2] の SDE 更新 (q_i[2]を q3 とする)
        S_C = external_inputs.get('Creativity_Score_SC', 0.0)
        q3_index = NUM_H_FACTORS - 1 # インデックス 2
        q3_current = self.state['q_i'][q3_index]
        
        # 駆動項: $\kappa_{\text{C}} \cdot \mathbf{S}_{\text{C}}(t) \cdot (1 - q_3/q_3^{\text{max}})$
        drift_increase = P['kappa_C'] * S_C * (1.0 - q3_current / P['q3_max'])
        
        # 減衰項: $-\rho_{\text{C}} \cdot q_3$
        drift_decay = - P['rho_C'] * q3_current
        
        # ノイズ項 (Z_aux[5] を使用)
        # dq3 = (drift_increase + drift_decay) dt + sigma_C dW_C(t)
        self.state['q_i'][q3_index] += (drift_increase + drift_decay) * dt + P['sigma_C'] * math.sqrt(dt) * Z_aux[5]
        
        # q3 が負にならないようクリッピング
        self.state['q_i'][q3_index] = max(0.0, self.state['q_i'][q3_index])


        # 変更点 4: 8. 瞬間貢献量の計算 (H_inst) - 二乗貢献の導入

        # $\mu_i$ ベクトル: $\mu_i = q_i r_i c_i v_i d_i$ (N_H=3)
        mu_i_vector = self.state['q_i'] * self.state['r_i'] * self.state['c_i'] * self.state['v_i'] * self.state['d_i']
        
        # f_i(t), s_i(t) の計算 (N_H=3)
        # w_i(t) を $q_i$ で近似。f_i(t), s_i(t) はここでは計算を簡略化している可能性あり。
        f_i = np.maximum(0, np.tanh(P['alpha_f'] * self.state['I']) / (1.0 + P['beta_f'] * self.state['H']))
        s_i = 1.0 / (1.0 + np.exp(-P['alpha_s'] * (self.state['eta_m'] - P['delta_s'])))

        # A. 創造性因子以外の線形貢献 (i=0, 1)
        # NUM_H_FACTORS - 1 は 2
        linear_contribution = np.sum(mu_i_vector[:NUM_H_FACTORS - 1] * f_i[:NUM_H_FACTORS - 1] * s_i[:NUM_H_FACTORS - 1])
        
        # B. 創造性因子 i=2 (q3) の二乗貢献
        # 貢献度: \mathbb{E}[w_3]\mathbb{E}[f_3]\mathbb{E}[s_3] に相当
        q3_contribution = mu_i_vector[q3_index] * f_i[q3_index] * s_i[q3_index] 
        
        # H^2 駆動: \lambda_C * (貢献度)^2
        quadratic_contribution = P['lambda_C'] * (q3_contribution ** 2)

        # 最終的なスカラー総和 $H_{\text{inst}}(t)$
        H_inst = linear_contribution + quadratic_contribution

        # $U_{\text{inst}}(t)$ の多次元総和 (変更なし)
        U_inst_factor_sum = np.sum(self.state['s_j'] + self.state['l_j'] + self.state['a_j'] +
                                   self.state['c_j'] + self.state['r_j'] + self.state['v_j'] + self.state['i_j'])
        U_inst = U_inst_factor_sum + U_inst_inter

        # 9. HALM記録と特異点判定 (変更なし)
        self.log_H_inst.append(H_inst)
        self.log_U_inst.append(U_inst)
        self.log_time.append(t)
        self.log_counter += 1
        if U_inst > P['theta_Trauma']:
            self.L_S.append({'t_start': t, 'U_inst': U_inst, 'H_inst': H_inst})

        # 10. 累積量 H(t), U(t) の更新 (変更なし)
        HALM_inf = self._calculate_halm_influence()
        dH_drift = (H_inst - P['beta_H'] * self.state['H'] + 0.0) + HALM_inf['H']
        dU_drift = (U_inst - P['beta'] * self.state['U'] + 0.0) + HALM_inf['U']

        self.state['H'] += dH_drift * dt
        self.state['U'] += dU_drift * dt

        # 11. 動機変数 $I(t)$ SDE の更新 (変更なし)
        dI_drift = P['alpha_I'] * (H_prime_prev - U_prime_prev) - P['gamma_I'] * self.state['I']
        self.state['I'] += dI_drift * dt + P['sigma_I'] * math.sqrt(dt) * Z_aux[2]

        # 12. 補正項 $P(t), R(t)$ の計算 (変更なし)
        Correction = self._calculate_correction_terms(self.state['H_env_avg'])
        P_t, R_t = Correction['P'], Correction['R']

        # 13. 最終 SDE のドリフト項 $\mathbf{\mu_P(\cdot)}, \mathbf{\mu_R(\cdot)}$ (変更なし)
        mu_P = (P_t - self.log_P_prev) / dt
        mu_R = (R_t - self.log_R_prev) / dt

        # 14. 最終 SDE の更新 ($\mathbf{\Delta H' = \kappa_H \Delta H + \Delta P}$ の厳密な離散化) (変更なし)
        dH_prime_drift = P['kappa_H'] * dH_drift + mu_P
        dU_prime_drift = P['kappa_U'] * dU_drift - mu_R

        H_prime_diff = P['sigma_H_prime'] * math.sqrt(dt) * Z_aux[3]
        U_prime_diff = P['sigma_U_prime'] * math.sqrt(dt) * Z_aux[4]

        jump_occur_H = self.rng.random() < P['lambda_H_prime'] * dt
        jump_size_H = self.rng.laplace(loc=0.0, scale=P['b_H_prime']) if jump_occur_H else 0.0
        jump_occur_U = self.rng.random() < P['lambda_H_prime'] * dt
        jump_size_U = self.rng.laplace(loc=0.0, scale=P['b_H_prime']) if jump_occur_U else 0.0

        H_prime = H_prime_prev + dH_prime_drift * dt + H_prime_diff + jump_size_H
        U_prime = U_prime_prev + dU_prime_drift * dt + U_prime_diff + jump_size_U

        # 15. 履歴 U_hist の更新 (変更なし)
        self.state['U_hist'] += self.state['U'] * dt

        # 16. HALM 抽象化とログ更新 (変更なし)
        self._consolidate_memory()
        self._abstract_longterm()

        # 17. 次ステップの計算のために P(t), R(t), U_env(t) の値を保持 (変更なし)
        self.log_P_prev = P_t
        self.log_R_prev = R_t
        self.log_U_env_prev = U_env

        # 18. 時刻のインクリメント
        self.t += dt

        return {'H_prime': H_prime, 'U_prime': U_prime, 't': self.t}
