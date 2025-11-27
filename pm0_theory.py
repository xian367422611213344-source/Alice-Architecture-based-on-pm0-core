import numpy as np
from typing import Dict, Any, List, Optional
import math


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

        # HALM/SDE 補助ロギング (質問 10, 13 で決定)
        self.log_U_env_prev: float = self.state['U_env_base']  # 衝撃駆動項 $\text{Impact}_j(t)$ 用
        self.log_P_prev: float = 0.0  # 最終 SDE ドリフト項 $\mu_P$ 用
        self.log_R_prev: float = 0.0  # 最終 SDE ドリフト項 $\mu_R$ 用

        # HALMカウンタ
        self.N_block = 10  # 1ブロックのステップ数 (1.0 day)
        self.N_consolidation = 5  # 統合に必要なブロック数 (5.0 day)
        self.log_counter = 0


    def _initialize_parameters(self) -> Dict[str, Any]:
        """全フェーズで確定したパラメータの初期設定を定義する。（簡略化はコード実行時のみ。定義は完全版）"""
        P = {}
        # I. HALM/忘却率 (質問 1-B, 9-3, 9-1)
        P.update({
            'beta_H': 0.50, 'beta': 0.50,  # H/U標準忘却率 (短期 SDE)
            'beta_Mid': 0.15, 'beta_Long': 0.01, 'beta_Singular': 0.001,
            'theta_Trauma': 3.0,  # 特異点判定閾値
        })
        # II. SDE係数と駆動パラメータ (質問 3, 9)
        P.update({
            # H因子 q, r, c, v, d (i依存性省略)
            'q_i0': 1.0, 'r_i0': 1.0, 'c_i0': 1.0, 'v_i0': 1.0, 'd_i0': 1.0,
            'alpha_i': 0.05, 'beta_i': 0.02, 'gamma_i': 0.01,  # q_i
            'alpha_r': 0.1, 'beta_r': 0.05,  # r_i
            'kappa_i': 0.10, 'rho_i': 0.05, 'theta_i': 1.0,  # d_i (f_past 係数 $\theta_i$)
            'delta_r': 0.1, 'delta_i': 0.1,  # match_event, isolation 感度
            # U因子 a, c, i (j依存性省略)
            's_j0': 1.0, 'l_j0': 1.0, 'a_j0': 1.0, 'c_j0': 1.0, 'r_j0': 1.0, 'v_j0': 1.0, 'i_j0': 1.0,
            'alpha_a': 0.1, 'gamma_a': 0.05,  # a_j ($\text{Recur}_j$)
            'alpha_c': 0.1, 'gamma_c': 0.05,  # c_j ($\text{Impact}_j$)
        })
        # III. c_i(t) 成長パラメータ (質問 3)
        P.update({
            'phi_max': 1.0, 'lambda_phi': 0.05,  # 柔軟性
            'psi_min': 0.1, 'psi_max': 1.0, 'k_psi': 0.2, 'age_1/2': 10.0,  # 安定性
        })
        # IV. 環境 SDE パラメータ (質問 2-C, 10, 12)
        P.update({
            'alpha_env': 0.01, 'sigma_env': 0.05, 'theta_env': 2.0,  # h_k SDE
            'lambda_k': 0.1, 'theta_k': 2.0,  # ポアソンジャンプ (ストレス)
            'alpha_avg': 0.2, 'sigma_avg': 0.05,  # 環境平均 SDE
            # $\tanh$ 非線形合成 (質問 12-1)
            'rho_m': 0.5, 'gamma_tilde_m': 0.5,  # H_env 係数
            'sigma_k_env': 0.5, 'gamma_k_env': 0.5,  # U_env 係数
            'U_env_base': 1.0, 'H_env_base': 1.0,  # 初期値用
        })
        # V. 較正/学習パラメータ (質問 8, 11, 12)
        P.update({
            'kappa_H': 1.0, 'kappa_U': 1.0,  # 較正ゲイン
            'alpha_lambda': 0.01, 'rho_lambda': 0.01, 'lambda_base': 0.0,  # 不幸相互作用 $\lambda_{jk}$
            'theta_R': 5.0, 'kappa_R': 1.0, 'lambda_R': 0.001,  # 回復/履歴減衰率
            'lambda_S': 0.1,  # トラウマ抑制率 $S_R(t)$
            # 意識的動機 $I(t)$ SDE (質問 10)
            'alpha_I': 0.05, 'gamma_I': 0.1, 'sigma_I': 0.1,
            # 瞬間幸福 $f_i(t), s_i(t)$ (質問 12)
            'alpha_f': 1.0, 'beta_f': 0.5, 'alpha_s': 2.0, 'delta_s': 0.0,
        })
        # VI. 補正項 P(t), R(t) の乗法要素 (質問 10)
        P.update({
            'gamma_P': 1.0, 'delta_P': 0.5, 'gamma_R': 1.0, 'delta_R': 0.5,  # A_P, A_R (環境依存)
            'epsilon_P': 0.1, 'epsilon_R': 0.1, 'T_day': 1.0, 'phi_P': 0.0, 'phi_R': math.pi,  # C_P, C_R (日周期)
            'alpha_P': 0.5, 'tau_P1': 10.0, 'tau_P2': 100.0,  # S_P (ライフフェーズ)
            'beta_P': 1.0, 'beta_R': 1.0,  # M_P, M_R (動機)
        })
        # VII. 最終 SDE / 認知ジャンプ (質問 6-C, 8-B)
        P.update({
            'sigma_H_prime': 0.1, 'sigma_U_prime': 0.1,  # 最終 SDE ノイズスケール
            'lambda_H_prime': 0.05, 'b_H_prime': 1.0,  # 認知ジャンプ (ラプラス分布)
        })
        # VIII. v_val(t) SDE (質問 7)
        P.update({
            'alpha_v': 0.05, 'T_trend': 365.0, 'A_v': 0.2, 'B_v': 1.0, 'sigma_v': 0.1,  # トレンド SDE
            'phi_v': 0.0,  # 位相
        })

        return P


    def _initialize_state(self) -> Dict[str, Any]:
        """状態変数の初期値を定義する。"""
        S = {'H': 0.0, 'U': 0.0}
        # 瞬間因子 (簡略化のためスカラー。実際は $N_H, N_U$ サイズ)
        S.update({'q_i': 1.0, 'r_i': 1.0, 'c_i': 1.0, 'v_i': 1.0, 'd_i': 1.0})
        S.update({'s_j': 1.0, 'l_j': 1.0, 'a_j': 1.0, 'c_j': 1.0, 'r_j': 1.0, 'v_j': 1.0, 'i_j': 1.0})
        # 環境 SDE (h_k: 不幸環境, eta_m: 幸福環境/社会性 $\eta_{\text{social}}$)
        S.update({'h_k': 1.0, 'eta_m': 1.0})
        # 補助 SDE / 履歴
        S.update({'v_val': self.params['B_v']})  # 価値観トレンド
        S.update({'U_hist': 0.0})  # 慢性ストレス履歴
        S.update({'I': 1.0})  # 意識的動機変数 I(t)
        S.update({'H_env_avg': self.params['H_env_base']})  # 平滑化された環境平均 $\overline{H}_{\text{env}}(t)$
        # $\Lambda$: 不幸相互作用ゲイン行列 (簡略化のため、ここではスカラー $\lambda_{sl}$ を使用)
        S.update({'lambda_sl': self.params['lambda_base']})
        S.update({'H_env_base': self.params['H_env_base'], 'U_env_base': self.params['U_env_base']})

        return S


    # --- 6. HALM 抽象化ロジック (質問 5) ---
    # _consolidate_memory と _abstract_longterm は、レビューコードの定義をそのまま採用。


    def _consolidate_memory(self):
        """短期ログを中期抽象化ブロックに圧縮し、L_Mに追加する。"""
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
        """中期ブロックを長期抽象ブロックに統合し、L_Mのサイズを抑制する。"""
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


    # --- 7. HALM 影響項の計算 ---
    # _calculate_halm_influence は、レビューコードの定義をそのまま採用。


    def _calculate_halm_influence(self) -> Dict[str, float]:
        """HALM記憶からの累積量への影響項 (µ^HALM の総和項) を計算する。"""
        H_influence = 0.0
        U_influence = 0.0

        # L_M 影響項 (レイヤー II/III: β_Long を使用)
        if self.L_M:
            t_m = np.array([b['t_start'] for b in self.L_M])
            H_mean = np.array([b['H_mean'] for b in self.L_M])
            U_mean = np.array([b['U_mean'] for b in self.L_M])
            k_M = exponential_decay_kernel(self.t, t_m, self.params['beta_Long'])
            H_influence += np.sum(H_mean * k_M)
            U_influence += np.sum(U_mean * k_M)

        # L_S 影響項 (レイヤー IV: β_Singular を使用)
        if self.L_S:
            t_s = np.array([s['t_start'] for s in self.L_S])
            H_inst_s = np.array([s['H_inst'] for s in self.L_S])
            U_inst_s = np.array([s['U_inst'] for s in self.L_S])
            k_S = exponential_decay_kernel(self.t, t_s, self.params['beta_Singular'])
            H_influence += np.sum(H_inst_s * k_S)
            U_influence += np.sum(U_inst_s * k_S)

        return {'H': H_influence, 'U': U_influence}


    # --- 8. 厳密な因子駆動項の計算 (質問 9 統合) ---


    def _calculate_driving_terms(self, H_env: float, U_env: float) -> Dict[str, float]:
        """質問 9 で確定した心理学的因子駆動項を計算する。"""
        P = self.params

        # 1. 慣れ駆動項 $f_{\text{past}}(t) \equiv E_{\text{past}}(t)$ (dd_i の駆動)
        # L_M の H 影響を $f_{\text{past}}(t)$ とする
        L_M_influence_H = 0.0
        if self.L_M:
            t_m = np.array([b['t_start'] for b in self.L_M])
            H_mean = np.array([b['H_mean'] for b in self.L_M])
            k_M_Long = exponential_decay_kernel(self.t, t_m, P['beta_Long'])
            L_M_influence_H = np.sum(H_mean * k_M_Long)
        f_past = L_M_influence_H

        # 2. イベント駆動項 $\text{match\_event}(t)$ (dr_i の駆動)
        match_event = 1.0 / (P['delta_r'] + abs(H_env - self.state['H']))

        # 3. 反復駆動項 $\text{Recur}_j(t)$ (da_j の駆動)
        recur_j = 0.0
        if self.L_M:
            t_m = np.array([b['t_start'] for b in self.L_M])
            U_var = np.array([b['U_var'] for b in self.L_M])
            k_M_Mid = exponential_decay_kernel(self.t, t_m, P['beta_Mid'])
            recur_j = np.sum(U_var * k_M_Mid)

        # 4. 衝撃駆動項 $\text{Impact}_j(t)$ (dc_j の駆動)
        # $\mathbf{|\Delta U_{\text{env}} / \Delta t|}$ の離散近似
        impact_j = abs(U_env - self.log_U_env_prev) / self.dt

        # 5. 孤立駆動項 $\text{Isolation}(t)$ (di_j の駆動)
        # $\eta_{\text{social}}$ はここでは $\eta_m$ と同じ値とする
        eta_social = self.state['eta_m']
        isolation = 1.0 / (P['delta_i'] + eta_social)

        return {
            'f_past': f_past, 'match_event': match_event, 'recur_j': recur_j,
            'impact_j': impact_j, 'isolation': isolation
        }


    # --- 9. 厳密な非線形補正・回復項の計算 (質問 10 統合) ---


    def _calculate_correction_terms(self, H_env_avg: float) -> Dict[str, float]:
        """P(t) (幸福ブースター) と R(t) (回復力) の完全な乗算構造を計算する。"""
        P = self.params
        t = self.t
        I = self.state['I']
        H = self.state['H']
        U = self.state['U']
        U_hist = self.state['U_hist']

        # --- P(t) の乗法要素 ---
        # A_P(t) (環境依存: $\overline{H}_{\text{env}}(t)$ に依存)
        A_P = 1.0 / (1.0 + math.exp(-P['gamma_P'] * (H_env_avg - P['delta_P'])))
        # C_P(t) (日周期)
        C_P = 1.0 + P['epsilon_P'] * math.cos(2 * math.pi / P['T_day'] * t + P['phi_P'])
        # S_P(t) (ライフフェーズ)
        S_P = P['alpha_P'] * math.exp(-t / P['tau_P1']) + (1 - P['alpha_P']) * math.exp(-t / P['tau_P2'])
        # M_P(t) (意識的動機)
        M_P = math.tanh(P['beta_P'] * I)

        P_t = A_P * C_P * S_P * M_P


        # --- R(t) の乗法要素 ---
        # A_R(t) (環境依存: $\overline{H}_{\text{env}}(t)$ に依存)
        A_R = 1.0 / (1.0 + math.exp(-P['gamma_R'] * (H_env_avg - P['delta_R'])))
        # C_R(t) (日周期: $\phi_R=\pi$)
        C_R = 1.0 + P['epsilon_R'] * math.cos(2 * math.pi / P['T_day'] * t + P['phi_R'])
        # T_R(t) (閾値効果/バーンアウト)
        T_R = 1.0 / (1.0 + math.exp(-P['kappa_R'] * (U - P['theta_R'])))
        # H_R(t) (履歴依存性: $\exp(-\lambda_R \int U d\tau)$)
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
        # M_R(t) (意識的動機)
        M_R = math.tanh(P['beta_R'] * I)

        R_t = A_R * C_R * T_R * H_R * S_R * M_R

        return {'P': P_t, 'R': R_t}


    # --- 10. メインステップ関数 (全 SDE/ODE 完全統合) ---


    def step(self, H_inst_input_base: float, U_inst_input_base: float):
        # 1. 時刻の更新と乱数生成
        dt, t = self.dt, self.t
        Z = self.rng.standard_normal(size=15)  # ノイズ項を増やす

        # 2. HALM記録と特異点判定
        self.log_H_inst.append(H_inst_input_base)
        self.log_U_inst.append(U_inst_input_base)
        self.log_time.append(t)
        self.log_counter += 1
        if U_inst_input_base > self.params['theta_Trauma']:
            self.L_S.append({'t_start': t, 'U_inst': U_inst_input_base, 'H_inst': H_inst_input_base})


        # 3. 補助 SDE の更新: v_val(t), h_k(t), H_env_avg(t)

        # v_val(t) SDE (質問 7 厳密版)
        m_v = self.params['A_v'] * math.sin(2 * math.pi / self.params['T_trend'] * t + self.params['phi_v']) + self.params['B_v']
        dv_val_drift = -self.params['alpha_v'] * (self.state['v_val'] - m_v)
        self.state['v_val'] += dv_val_drift * dt + self.params['sigma_v'] * math.sqrt(dt) * Z[0]


        # h_k(t) SDE (ストレス環境 - 質問 2 厳密版)
        dh_k_drift = -self.params['alpha_env'] * (self.state['h_k'] - 1.0)  # 平均回帰ターゲットを 1.0 と仮定
        jump_occur_k = self.rng.random() < self.params['lambda_k'] * dt
        jump_size_k = self.rng.exponential(scale=1.0 / self.params['theta_k']) if jump_occur_k else 0.0
        self.state['h_k'] += dh_k_drift * dt + self.params['sigma_env'] * math.sqrt(dt) * Z[1] + jump_size_k

        # eta_m(t) (幸福環境/社会性 - 簡略化のため定常としておく)
        self.state['eta_m'] = 1.0

        # 環境項の非線形合成 (質問 12-1 厳密版)
        H_env = self.params['rho_m'] * math.tanh(self.params['gamma_tilde_m'] * self.state['eta_m'])
        U_env = self.params['sigma_k_env'] * math.tanh(self.params['gamma_k_env'] * self.state['h_k'])

        # $\overline{H}_{\text{env}}(t)$ SDE (質問 10-1 厳密版)
        dH_env_avg_drift = self.params['alpha_avg'] * (H_env - self.state['H_env_avg'])
        self.state['H_env_avg'] += dH_env_avg_drift * dt + self.params['sigma_avg'] * math.sqrt(dt) * Z[2]

        # 4. 前ステップ値の準備
        H_prime_prev = self.params['kappa_H'] * self.state['H'] + self.log_P_prev
        U_prime_prev = self.params['kappa_U'] * self.state['U'] - self.log_R_prev

        # 5. 不幸相互作用項 $\lambda_{jk}$ の更新 (質問 11-1 厳密版)
        lambda_sl = self.state['lambda_sl']
        nu_s = self.state['s_j']; nu_l = self.state['l_j']  # 不幸因子 s_j, l_j と仮定
        d_lambda_drift = self.params['alpha_lambda'] * nu_s * nu_l - self.params['rho_lambda'] * (lambda_sl - self.params['lambda_base'])
        self.state['lambda_sl'] += d_lambda_drift * dt
        # 相互作用項の総和 $\mathbf{U_{\text{inst}, \text{inter}}}$ (質問 11-2 厳密版)
        U_inst_inter = 2 * self.state['lambda_sl'] * nu_s * nu_l  # j!=k の二重総和 (スカラー近似)

        # 6. 厳密な因子駆動項の計算 (質問 9)
        Driving = self._calculate_driving_terms(H_env, U_env)
        f_past = Driving['f_past']; match_event = Driving['match_event']; recur_j = Driving['recur_j']
        impact_j = Driving['impact_j']; isolation = Driving['isolation']


        # 7. 瞬間因子 SDE の更新 (質問 9 厳密版)
        # H因子 d_i(t) の更新 (慣れ - $f_{\text{past}}$)
        dd_i_drift = -self.params['kappa_i'] * f_past - self.params['rho_i'] * (self.state['d_i'] - self.params['d_i0'])  # $f_{\text{past}}$ の係数を $\kappa_i$ に変更
        self.state['d_i'] += dd_i_drift * dt + self.params['sigma_env'] * math.sqrt(dt) * Z[3]
        # H因子 r_i(t) の更新 (整合性 - $\text{match\_event}$)
        dr_i_drift = self.params['alpha_r'] * match_event - self.params['beta_r'] * (self.state['r_i'] - self.params['r_i0'])
        self.state['r_i'] += dr_i_drift * dt + self.params['sigma_env'] * math.sqrt(dt) * Z[4]
        # U因子 a_j(t) の更新 (不安 - $\text{Recur}_j$)
        da_j_drift = self.params['alpha_a'] * recur_j - self.params['gamma_a'] * (self.state['a_j'] - self.params['a_j0'])
        self.state['a_j'] += da_j_drift * dt + self.params['sigma_env'] * math.sqrt(dt) * Z[5]
        # U因子 c_j(t) の更新 (脆弱性 - $\text{Impact}_j$)
        dc_j_drift = self.params['alpha_c'] * impact_j - self.params['gamma_c'] * (self.state['c_j'] - self.params['c_j0'])
        self.state['c_j'] += dc_j_drift * dt + self.params['sigma_env'] * math.sqrt(dt) * Z[6]
        # U因子 i_j(t) の更新 (孤立 - $\text{Isolation}$)
        di_j_drift = 0.1 * isolation - 0.05 * (self.state['i_j'] - self.params['i_j0'])
        self.state['i_j'] += di_j_drift * dt + self.params['sigma_env'] * math.sqrt(dt) * Z[7]
        # c_i(t) の更新 (時間依存性 $\phi, \psi$ を使用 - 質問 3 厳密版)
        phi = phi_age(t, self.params)
        psi = psi_age(t, self.params)
        dc_i_drift = phi * self.state['v_val'] - psi * (self.state['c_i'] - self.params['c_i0'])
        self.state['c_i'] += dc_i_drift * dt + self.params['sigma_env'] * math.sqrt(dt) * Z[8]


        # 8. 瞬間貢献量の計算 (H_inst, U_inst)

        # 瞬間幸福 $H_{\text{inst}}(t)$ の完全総和構造 (質問 12-2 厳密版)
        mu_i = 1.0 * self.state['q_i'] * self.state['r_i'] * self.state['c_i'] * self.state['v_i'] * self.state['d_i']
        # $f_i(t)$ (行動動機: I(t)とH(t)に依存)
        f_i = max(0, math.tanh(self.params['alpha_f'] * self.state['I']) / (1.0 + self.params['beta_f'] * self.state['H']))
        # $s_i(t)$ (社会性: $\eta_{\text{social}}$ に依存)
        s_i = 1.0 / (1.0 + math.exp(-self.params['alpha_s'] * (self.state['eta_m'] - self.params['delta_s'])))

        H_inst = mu_i * f_i * s_i  # 簡略化のため i=1 のみ

        # 瞬間不幸 $U_{\text{inst}}(t)$
        U_inst_factor_sum = 1.0 * (self.state['s_j'] + self.state['l_j'] + self.state['a_j'] + self.state['c_j'] + self.state['r_j'] + self.state['v_j'] + self.state['i_j'])
        U_inst = U_inst_factor_sum + U_inst_inter

        # 9. 累積量 H(t), U(t) の更新 (HALM $\mathbf{\mu^{\text{HALM}}}$ 統合)
        HALM_inf = self._calculate_halm_influence()
        dH_drift = (H_inst - self.params['beta_H'] * self.state['H'] + 0.0) + HALM_inf['H']
        dU_drift = (U_inst - self.params['beta'] * self.state['U'] + 0.0) + HALM_inf['U']

        self.state['H'] += dH_drift * dt
        self.state['U'] += dU_drift * dt

        # 10. 動機変数 $I(t)$ SDE の更新 (質問 10-4 厳密版)
        dI_drift = self.params['alpha_I'] * (H_prime_prev - U_prime_prev) - self.params['gamma_I'] * self.state['I']
        self.state['I'] += dI_drift * dt + self.params['sigma_I'] * math.sqrt(dt) * Z[9]

        # 11. 補正項 $P(t), R(t)$ の計算 (質問 10, 13)
        Correction = self._calculate_correction_terms(self.state['H_env_avg'])
        P_t, R_t = Correction['P'], Correction['R']

        # 12. 最終 SDE のドリフト項 $\mathbf{\mu_P(\cdot)}, \mathbf{\mu_R(\cdot)}$ (質問 13 厳密版)
        mu_P = (P_t - self.log_P_prev) / dt
        mu_R = (R_t - self.log_R_prev) / dt

        # 13. 最終 SDE の更新 ($\mathbf{\Delta H' = \kappa_H \Delta H + \Delta P}$ の厳密な離散化)

        # ドリフト項: $\kappa_H \cdot \mathbf{\mu_{H}^{\text{HALM}}} + \mathbf{\mu_P}$
        dH_prime_drift = self.params['kappa_H'] * dH_drift + mu_P
        # ドリフト項: $\kappa_U \cdot \mathbf{\mu_{U}^{\text{HALM}}} - \mathbf{\mu_R}$
        dU_prime_drift = self.params['kappa_U'] * dU_drift - mu_R

        # 最終状態のノイズ項とジャンプ項 (質問 6 厳密版)
        H_prime_diff = self.params['sigma_H_prime'] * math.sqrt(dt) * Z[10]
        U_prime_diff = self.params['sigma_U_prime'] * math.sqrt(dt) * Z[11]

        jump_occur_H = self.rng.random() < self.params['lambda_H_prime'] * dt
        jump_size_H = self.rng.laplace(loc=0.0, scale=self.params['b_H_prime']) if jump_occur_H else 0.0
        jump_occur_U = self.rng.random() < self.params['lambda_H_prime'] * dt  # $\lambda_{U'}$ も $\lambda_{H'}$ と同じと仮定
        jump_size_U = self.rng.laplace(loc=0.0, scale=self.params['b_H_prime']) if jump_occur_U else 0.0

        H_prime = H_prime_prev + dH_prime_drift * dt + H_prime_diff + jump_size_H
        U_prime = U_prime_prev + dU_prime_drift * dt + U_prime_diff + jump_size_U


        # 14. 履歴 U_hist の更新 (慢性ストレス積分項のリーマン和近似)
        self.state['U_hist'] += self.state['U'] * dt

        # 15. HALM 抽象化とログ更新
        self._consolidate_memory()
        self._abstract_longterm()

        # 16. 次ステップの計算のために P(t), R(t), U_env(t) の値を保持
        self.log_P_prev = P_t
        self.log_R_prev = R_t
        self.log_U_env_prev = U_env

        # 17. 時刻のインクリメント
        self.t += dt

        return {'H_prime': H_prime, 'U_prime': U_prime, 't': self.t}
