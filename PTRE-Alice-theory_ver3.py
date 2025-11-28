import numpy as np
import copy
import logging

# ロギング設定 (デバッグ用)
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')


# --- 1. CORE CONFIGURATION AND HYPERPARAMETERS (PTRE統合) ---

class Config:
    """Alice Architectureの全ハイパーパラメータと次元設定"""

    # 全体の次元設定
    N_C = 64        # 意識（意味構造）層 C の次元
    N_M = 32        # 記憶（情動/エピソード）層 M の次元
    N_H = 16        # 幸福（快/不快）核 H の次元
    N_R = 8         # 報酬予測層 R の次元
    N_ESELF = 16    # 自己表象予測層 E_self_pred の次元
    N_E_P = 32      # 環境入力 E_env の予測層 P の次元
    N_ACTION = 1    # 行動 A の次元

    # BPTT (Backpropagation Through Time) 学習設定
    T_BPTT = 16     # BPTTの遡及時間窓 (TDLの履歴長)
    ETA_TDL = 1e-4  # A-TDL (スキル学習) の学習率
    GAMMA = 0.99    # 割引率

    # CSC (Conscious Stabilization Condition) ハイパーパラメータ
    CSC_MAX_ITER = 50       # CSCの最大反復回数 (努力 K の上限として使用)
    CSC_TOLERANCE = 1e-4    # 収束許容誤差
    CSC_ETA = 0.05          # C層の固定点探索時の学習率 (PTRE版で使用)

    # V_total (価値関数) のハイパーパラメータ (旧式コスト項の感応度)
    LAMBDA_P = 0.5      # 予測誤差 P のコスト感応度
    LAMBDA_C = 0.2      # 制御負荷 Var(E_ctrl) のコスト感応度
    LAMBDA_S = 0.1      # 自己不整合 Dist_self のコスト感応度

    # PTRE 人格パラメータ theta の上限と初期値
    THETA_MAX = 1.0     # 人格パラメータのクリッピング上限
    # 努力コスト感応度 kappa_K
    THETA_KAPPA_INIT = 0.01
    # 努力コスト指数 beta_K (1.0より大きく設定)
    THETA_BETA_INIT = 1.2
    # K抑制感応度 gamma_K (Kが大きい時の進化抑制度)
    THETA_GAMMA_K_INIT = 0.3

    # 人格進化則ハイパーパラメータ
    ETA_THETA_BASE = 1e-6    # 人格進化の基本学習率

    # ノイズのレベル
    NOISE_LEVEL = 0.01

    def __init__(self):
        # E_envの次元を動的に計算 (E_tokenはN_Cと同じ次元と仮定, E_scalar=4)
        self.N_E_ENV = self.N_C + self.N_ESELF + 4


# --- 2. A-TDL (Autonomous Temporal Difference Learning) Learner ---

class ATDLLearner:
    """スキル（構造）学習則を実行するモジュール。BPTTとTD誤差を使用。"""
    def __init__(self, config: Config):
        self.config = config
        self.history = []  # 履歴 [(H, U, R, E_ctrl, C, M, E_env, A)]

    def _apply_gradient(self, delta_W, W):
        """勾配の適用とクリッピング (簡易版)"""
        # 簡易勾配降下法を想定
        # 実際には、クリッピングや最適化アルゴリズムをここに実装可能
        pass
        return W + self.config.ETA_TDL * delta_W

    def learn_step(self, V_current: float, V_prev: float, prev_state, theta: dict) -> float:
        """
        V(t)とV(t-1)を使用してTD誤差を計算する。
        """
        cfg = self.config

        # 1. TD誤差の計算 (R(t-1) + gamma * V(t) - V(t-1))
        R_prev = prev_state['R']
        TD_error = R_prev[0] + cfg.GAMMA * V_current - V_prev

        # 2. BPTTのための学習率変調 (TD誤差の絶対値に基づく)
        # 不安 U_pz の平均値が低いほど、学習に自信を持つ（学習率を上げる）
        U_pz_mean = np.mean(prev_state['U_pz'])
        theta_gamma_U = theta.get('theta_gamma_K', cfg.THETA_GAMMA_K_INIT)

        # 不安による学習抑制
        modulated_eta = cfg.ETA_TDL * np.exp(-theta_gamma_U * U_pz_mean)

        # 3. W_C の更新に必要な情報を計算（実際の更新はAliceArchitectureで行う）
        C_prev = prev_state['C']
        # Net_Cは_update_layersで計算されたもの
        C_current_projection = np.tanh(prev_state['Net_C'])

        # dW_C の簡易勾配: TD_error * C_current_projection * C_prev.T
        # delta_W_C = TD_error * np.outer(C_current_projection, C_prev)

        return TD_error


# --- 3. Alice Architecture Core Class (F_total) ---

class AliceArchitecture:
    """PTRE F_total に基づくAlice Architecture V3.0のコア実装"""
    def __init__(self):
        self.config = Config()
        cfg = self.config
        self.t = 0
        self.external_reward = 0.0

        # --- 状態変数 (Current State) ---
        self.C = np.zeros((cfg.N_C, 1))         # 意識（意味構造）層
        self.M = np.zeros((cfg.N_M, 1))         # 記憶層
        self.H = np.zeros((cfg.N_H, 1))         # 幸福核
        self.U = np.zeros((cfg.N_H, 1))         # 不安核
        self.R = np.zeros((cfg.N_R, 1))         # 報酬予測
        self.E_ctrl = np.zeros((cfg.N_C, 1))    # 制御予測誤差
        self.E_self = np.zeros((cfg.N_ESELF, 1)) # 自己表象予測誤差

        self.H_pz = np.zeros((cfg.N_H, 1))      # 安定化されたH (情動核の陽性/ゼロ)
        self.U_pz = np.zeros((cfg.N_H, 1))      # 安定化されたU (情動核の陽性/ゼロ)
        self.P = np.zeros((cfg.N_E_P, 1))       # 環境予測誤差
        self.VFL = np.zeros((cfg.N_R, 1))       # 価値予測学習項
        self.U_prime_pz = np.zeros((cfg.N_H, 1)) # 情動学習項
        self.E_self_pred = np.zeros((cfg.N_ESELF, 1)) # 自己表象の予測値

        # --- Next State (CSC/update_layers用の一時バッファ) ---
        # _next状態をバッファとして保持 (初期状態のコピー)
        self.C_next = self.C.copy()
        self.M_next = self.M.copy()
        self.H_next = self.H.copy()
        self.U_next = self.U.copy()
        self.R_next = self.R.copy()
        self.E_ctrl_next = self.E_ctrl.copy()
        self.E_self_next = self.E_self.copy()
        self.H_pz_next = self.H_pz.copy()
        self.U_pz_next = self.U_pz.copy()
        self.P_next = self.P.copy()
        self.VFL_next = self.VFL.copy()
        self.U_prime_pz_next = self.U_prime_pz.copy()
        self.E_self_pred_next = self.E_self_pred.copy()

        # Netの保持 (CSCロジックで使用)
        self.Net_C = np.zeros((cfg.N_C, 1))
        self.Net_A_next = np.zeros((cfg.N_ACTION, 1))

        # --- 結合重み (W: Recurrent, U: Input, B: Bias) ---
        # C層
        self.W_C = np.random.randn(cfg.N_C, cfg.N_C) * 0.1
        self.U_C_Eenv = np.random.randn(cfg.N_C, cfg.N_E_ENV) * 0.1
        self.U_C_M = np.random.randn(cfg.N_C, cfg.N_M) * 0.1
        self.b_C = np.zeros((cfg.N_C, 1))
        # M層
        self.U_M_C = np.random.randn(cfg.N_M, cfg.N_C) * 0.1
        # H, U層
        self.U_H_R = np.random.randn(cfg.N_H, cfg.N_R) * 0.1
        self.U_H_U = np.random.randn(cfg.N_H, cfg.N_H) * 0.1
        self.U_H_C = np.random.randn(cfg.N_H, cfg.N_C) * 0.1
        self.U_U_E = np.random.randn(cfg.N_H, cfg.N_C) * 0.1      # 制御誤差の寄与 (E_ctrl)
        self.U_U_S = np.random.randn(cfg.N_H, cfg.N_ESELF) * 0.1  # 自己予測誤差の寄与 (E_self)
        self.b_H = np.zeros((cfg.N_H, 1))
        # R層 (報酬予測)
        self.U_R_C = np.random.randn(cfg.N_R, cfg.N_C) * 0.1
        # 誤差予測層 (E_ctrl, P, E_self)
        self.U_E_C = np.random.randn(cfg.N_C, cfg.N_C) * 0.1      # E_ctrlの自己再帰
        self.U_P_C = np.random.randn(cfg.N_E_P, cfg.N_C) * 0.1
        self.U_Eself_C = np.random.randn(cfg.N_ESELF, cfg.N_C) * 0.1
        # 行動出力層 (NLG)
        self.U_NLG_C = np.random.randn(cfg.N_ACTION, cfg.N_C) * 0.1
        self.U_NLG_M = np.random.randn(cfg.N_ACTION, cfg.N_M) * 0.1

        # --- 人格パラメータ theta (PTRE版) ---
        self.theta = {
            # PTRE 努力コスト関連
            'theta_kappa': cfg.THETA_KAPPA_INIT,    # 努力コスト感応度 $\kappa$
            'theta_beta': cfg.THETA_BETA_INIT,      # 努力コスト指数 $\beta$
            'theta_gamma_K': cfg.THETA_GAMMA_K_INIT,# K抑制感応度 $\gamma_K$ (進化率変調用)

            # 旧式情動パラメータ (TDL/VFLの学習率変調に使用)
            'alpha_H': 0.1, 'beta_H': 0.1,
            'alpha_U': 0.1, 'beta_U': 0.1,
            'gamma_HU': 0.1, 'gamma_UH': 0.1,
            'kappa_U': 0.1,
            'H_base': 0.5
        }

        # --- 学習モジュール ---
        self.learner = ATDLLearner(cfg)
        self.V_prev = 0.0 # V(t-1)の値を格納 (TDL用)


    def _add_noise(self, size: int):
        """ノイズの付加（行動選択ノイズ、情動核ノイズなど）"""
        return np.random.randn(size, 1) * self.config.NOISE_LEVEL

    def _f_Will(self, H, R):
        """Will信号 (意図の強さ) の計算 (簡易版)"""
        # HとRの平均に基づいて意志決定の強さをモデル化
        return np.tanh(np.mean(H) + np.mean(R))

    def _generate_E_token(self, text: str):
        """
        環境入力 E_token の生成 (テキストを埋め込みベクトルに変換 - 簡易版)
        """
        cfg = self.config

        if not text:
            # 継続的な観測をシミュレート
            base_vec = np.sin(self.t / 10) * np.ones((cfg.N_C, 1))
        else:
            # 意味を持つ入力として、ランダムなノイズを付加
            np.random.seed(len(text) + self.t)
            base_vec = np.random.randn(cfg.N_C, 1) * 0.5

        return np.tanh(base_vec)

    def _save_history(self, E_env_t, A_t_initial):
        """現在の状態を履歴に保存 (BPTT/TDL用)"""

        history_item = {
            't': self.t,
            'C': self.C.copy(),
            'M': self.M.copy(),
            'H': self.H.copy(),
            'U': self.U.copy(),
            'R': self.R.copy(),
            'E_ctrl': self.E_ctrl.copy(),
            'E_self': self.E_self.copy(),
            'H_pz': self.H_pz.copy(),
            'U_pz': self.U_pz.copy(),
            'E_env': E_env_t.copy(),
            'A': A_t_initial,
            'Net_C': self.Net_C.copy() # _update_layersで計算されたものを保存
        }
        self.learner.history.append(history_item)
        if len(self.learner.history) > self.config.T_BPTT:
            self.learner.history.pop(0)

    def _update_layers(self, E_env_t, A_t):
        """
        F_total (全再帰写像) を実行し、次のタイムステップの状態を暫定的に計算する。
        これはCSC前のベース計算であり、CSCではこの後の反復更新を行う。
        """
        cfg = self.config
        theta = self.theta

        # 1. 報酬予測層 R の更新
        Net_R = self.U_R_C @ self.C
        self.R_next = (1 - theta['beta_H']) * self.R + theta['beta_H'] * np.tanh(Net_R)
        self.VFL_next = self.R_next * self.external_reward # VFL = R * E_ext

        # 2. 情動核 H, U の更新
        # RによるHの駆動
        H_R_drive = self.U_H_R @ self.R
        # CによるH, Uの相互駆動
        H_C_drive = self.U_H_C @ self.C

        Net_H = (1 - theta['gamma_HU']) * self.H + theta['alpha_H'] * H_R_drive + H_C_drive + self.b_H
        Net_U = (1 - theta['gamma_UH']) * self.U + theta['alpha_U'] * self.U_U_E @ self.E_ctrl + self.b_H

        self.H_next = np.clip(np.tanh(Net_H), -cfg.THETA_MAX, cfg.THETA_MAX)
        self.U_next = np.clip(np.tanh(Net_U), -cfg.THETA_MAX, cfg.THETA_MAX)

        # 情動核の陽性/ゼロ項 (安定化ロジック)
        self.H_pz_next = np.clip(self.H_next + theta['H_base'], 0, 1)
        self.U_pz_next = np.clip(self.U_next + theta['H_base'], 0, 1)

        # 3. 環境予測誤差 P の更新
        P_pred_E = self.U_P_C @ self.C # Cに基づくE_envの予測
        self.P_next = E_env_t - P_pred_E # 誤差

        # 4. 自己表象予測誤差 E_self の更新
        E_self_pred = self.U_Eself_C @ self.C
        self.E_self_pred_next = E_self_pred # 予測値を保存
        self.E_self_next = self.E_self - E_self_pred # 誤差

        # 5. 制御予測誤差 E_ctrl の更新
        C_pred_ctrl = self.U_E_C @ self.C # Cに基づく次のCの予測
        self.E_ctrl_next = np.tanh(C_pred_ctrl) - self.C # 制御誤差

        # 6. 記憶層 M の更新
        Net_M = self.U_M_C @ self.C
        self.M_next = (1 - theta['alpha_H']) * self.M + theta['alpha_H'] * np.tanh(Net_M)

        # 7. 意識層 C の更新 (次のCの暫定値)
        # CSCループに入る前のベース Net_C を保持
        Net_C = (self.W_C @ self.C + self.U_C_Eenv @ E_env_t +
                 self.U_C_M @ self.M + self.b_C)
        self.Net_C = Net_C # CSCで反復更新するためのベースNet
        self.C_next = np.tanh(Net_C)


    def _commit_state(self):
        """
        意識的安定化条件 (CSC) が完了した後、_next状態をcurrent状態にコミットする。
        """
        self.C = self.C_next
        self.M = self.M_next
        self.H = self.H_next
        self.U = self.U_next
        self.R = self.R_next
        self.E_ctrl = self.E_ctrl_next
        self.E_self = self.E_self_next
        self.H_pz = self.H_pz_next
        self.U_pz = self.U_pz_next
        self.P = self.P_next
        self.VFL = self.VFL_next
        self.U_prime_pz = self.U_prime_pz_next
        self.E_self_pred = self.E_self_pred_next


    def _run_csc_stabilization(self, E_env_t, A_t_initial):
        """
        [PTRE F_total 統合] 厳密な意識的安定化条件 (CSC) 反復と努力 K の計測。
        固定点探索によりCとMを収束させ、反復回数から努力 K を計算する。
        """
        cfg = self.config
        theta = self.theta

        # 安定化前の現在の状態 C, M をコピー
        C_k, M_k = self.C.copy(), self.M.copy()
        H_k, U_k = self.H_pz.copy(), self.U_pz.copy()

        K = 0 # 努力カウンター
        eta_CSC = cfg.CSC_ETA

        while K < cfg.CSC_MAX_ITER:
            K += 1

            # --- C層の固定点探索 ---
            # 情動バイアス: (H_k - U_k) の平均値に基づく
            emotion_bias = (np.mean(H_k) - np.mean(U_k)) * theta['kappa_U']

            # 1. 意味構造層 C の固定点探索
            Net_C_k = (self.W_C @ C_k + self.U_C_Eenv @ E_env_t +
                       self.U_C_M @ M_k + self.b_C + emotion_bias)

            # 勾配降下的な更新 (収束を確実にするためのSoft Update)
            C_next = (1 - eta_CSC) * C_k + eta_CSC * np.tanh(Net_C_k)

            # 2. 記憶層 M の更新
            alpha_M = theta['alpha_H'] # 記憶層の学習率として使用
            Net_M_k = self.U_M_C @ C_k
            M_next = (1 - alpha_M) * M_k + alpha_M * np.tanh(Net_M_k)

            # 3. 情動核の微小動的更新 (安定化を支援 - 簡略版)
            H_k = (1 - 0.01) * H_k + 0.01 * (np.mean(H_k) + 0.5)
            U_k = (1 - 0.01) * U_k + 0.01 * (np.mean(U_k) + 0.5)

            # 4. 収束判定 (C層とM層)
            if np.linalg.norm(C_next - C_k) < cfg.CSC_TOLERANCE and \
               np.linalg.norm(M_next - M_k) < cfg.CSC_TOLERANCE:
                break

            C_k, M_k = C_next, M_next

        # 安定化後のC, M, H_pz, U_pzを_next状態としてセット
        self.C_next = C_k
        self.M_next = M_k
        self.H_pz_next = H_k
        self.U_pz_next = U_k

        # 行動の最終決定 (CSC後の C_next, M_next に基づく)
        Net_A_next = self.U_NLG_C @ C_k + self.U_NLG_M @ M_k
        # ノイズの付加
        A_final_refined = np.tanh(Net_A_next)[0] + self._add_noise(1)[0]
        self.Net_A_next = Net_A_next

        # 努力 K (反復回数) と最終行動を返す
        return np.clip(A_final_refined, -1.0, 1.0), K


    def _calculate_V_from_state(self, K: int, state_prefix: str):
        """
        [PTRE F_total 統合] 努力 K に基づく V_new (総価値) の計算。
        $V_{\text{new}} = V_{\text{base}} - \text{Effort\_Cost}$ ($\theta^\kappa K^{\theta^\beta}$)
        """
        cfg = self.config

        # 1. V_base (旧コードのV_total) の計算に使用する安定化された状態の値を取得
        VFL = getattr(self, state_prefix + 'VFL')
        U_prime_pz = getattr(self, state_prefix + 'U_prime_pz')
        P = getattr(self, state_prefix + 'P')
        E_ctrl = getattr(self, state_prefix + 'E_ctrl')
        E_self = getattr(self, state_prefix + 'E_self')
        E_self_pred = getattr(self, state_prefix + 'E_self_pred')
        H = getattr(self, state_prefix + 'H')
        R = getattr(self, state_prefix + 'R')

        V_value = np.sum(VFL) # ポジティブな価値項

        # 旧式のコスト計算 (V_baseの一部)
        # 予測誤差 P コスト
        max_U_prime = np.max(U_prime_pz)
        V_affect_P = cfg.LAMBDA_P * np.sum(P**2) * (1.0 + self.theta['kappa_U'] * max_U_prime)
        # 制御負荷 Var(E_ctrl) コスト
        Var_ctrl = np.var(E_ctrl)
        V_affect_C = cfg.LAMBDA_C * Var_ctrl
        # 自己不整合 Dist_self コスト
        V_coherence = cfg.LAMBDA_S * np.sum((E_self - E_self_pred)**2)

        V_base = V_value - V_affect_P - V_affect_C - V_coherence

        # 2. 努力コストの計算 (PTRE法則の適用)
        theta_kappa = self.theta.get('theta_kappa', cfg.THETA_KAPPA_INIT)
        theta_beta = self.theta.get('theta_beta', cfg.THETA_BETA_INIT)

        # Effort Cost = theta_kappa * K^theta_beta
        effort_cost = theta_kappa * (K ** theta_beta)

        # V_new (V_total) = V_base - Effort_Cost
        V_new = V_base - effort_cost

        # 価値項の詳細 (ロギング用)
        V_terms = {
            'V_base': V_base,
            'Effort_Cost': effort_cost,
            'Dist_self': V_coherence,
            'Var_ctrl': Var_ctrl,
            'sum_VFL': V_value,
            'Will_signal': self._f_Will(H, R)
        }
        return V_new, V_terms


    def _evolve_theta(self, TD_error: float, K: int):
        """
        [PTRE F_total 統合] 努力 K と不安 U_pz に基づく人格進化則。
        TD誤差と安定化 K に応じて $\theta^\kappa$ と $\theta^{\gamma_K}$ を更新。
        """
        cfg = self.config
        theta = self.theta

        # 1. 人格パラメータの取得
        theta_kappa = theta.get('theta_kappa', cfg.THETA_KAPPA_INIT)
        theta_gamma_K = theta.get('theta_gamma_K', cfg.THETA_GAMMA_K_INIT)

        # 2. U_pzとKに基づく自律的学習率の変調
        U_pz_mean = np.mean(self.U_pz)
        K_max = cfg.CSC_MAX_ITER

        # Kによる抑制 (指数関数的減衰)
        eta_theta_K = cfg.ETA_THETA_BASE * np.exp(-theta_gamma_K * K / K_max)

        # 安定度・不安による更新抑制 (S_stability と S_U)
        S_stability = max(0.0, 1.0 - K / K_max) # Kが大きい（不安定）ならS_stabilityは0に近い
        S_U = max(0.0, 1.0 - U_pz_mean)          # U_pzが高い（不安）ならS_Uは0に近い

        # 総合進化シグナル
        update_magnitude = eta_theta_K * S_stability * S_U
        sign_TD = np.sign(TD_error)

        # 3. パラメータ更新 (TD_errorの符号と安定度に依存)
        # A. theta_kappa (努力コスト感応度) の更新: TD_errorを打ち消す方向
        # TD_errorが正 (報酬が期待以上) かつKが小さい -> theta_kappaを小さくする (努力を許容)
        # TD_errorが負 (報酬が期待以下) かつKが大きい -> theta_kappaを大きくする (努力を嫌う)
        delta_kappa = -update_magnitude * sign_TD * (K / K_max) * theta_kappa * 0.1

        # B. theta_gamma_K (K抑制感応度) の更新:
        delta_gamma_K = update_magnitude * np.abs(TD_error) * (K / K_max) * 0.1

        # $\theta$ベクトルの更新
        theta['theta_kappa'] = np.clip(theta_kappa + delta_kappa, 0.001, cfg.THETA_MAX)
        theta['theta_gamma_K'] = np.clip(theta_gamma_K + delta_gamma_K, 0.001, cfg.THETA_MAX)

        # Delta_Theta_Normの計算のために、ここではdelta_kappaの絶対値を返す
        return np.abs(delta_kappa)


    def _apply_tdl_gradients(self, TD_error: float, history_item: dict):
        """
        ATDLLearnerで計算されたTD誤差を用いて、実際に重みを更新する。
        ここではW_Cのみを更新対象とする（簡略化）。
        """
        cfg = self.config

        # 不安による学習抑制 (ATDLLearner内と同じロジックを使用)
        U_pz_mean = np.mean(history_item['U_pz'])
        theta_gamma_U = self.theta.get('theta_gamma_K', cfg.THETA_GAMMA_K_INIT)
        modulated_eta = cfg.ETA_TDL * np.exp(-theta_gamma_U * U_pz_mean)

        # 簡易BPTTの勾配計算 (C(t-1)とC(t)の再帰ループ)
        C_prev = history_item['C']
        C_current_projection = np.tanh(history_item['Net_C'])

        # dW_C の簡易勾配: TD_error * C_current_projection * C_prev.T
        delta_W_C = TD_error * np.outer(C_current_projection, C_prev)

        # W_C の更新
        W_C_next = self.W_C + modulated_eta * delta_W_C
        self.W_C = W_C_next

        return np.linalg.norm(delta_W_C) # スキル学習の変動ノルム


    def step(self, user_input_text: str, external_reward: float):
        """
        Alice Architecture の単一タイムステップを実行する (最終統合版)。
        """
        self.t += 1
        self.external_reward = external_reward
        cfg = self.config

        # --- 1. 環境入力 E_env の生成 ---
        E_token = self._generate_E_token(user_input_text)
        E_context = np.random.randn(cfg.N_ESELF) * 0.1
        E_scalar = np.array([external_reward, self.t % 100, self.t, 1 if np.random.rand() < 0.1 else 0]).reshape(-1, 1)
        E_env_t = np.concatenate([E_token, E_context.reshape(-1, 1), E_scalar])

        # --- 2. 初期行動 A(t) の生成 (tの状態C, Mに基づく) ---
        Net_A_t = self.U_NLG_C @ self.C + self.U_NLG_M @ self.M
        A_t_initial = np.tanh(Net_A_t)[0] + self._add_noise(1)[0]
        A_t_initial = np.clip(A_t_initial, -1.0, 1.0)

        # --- 3. F_total (全再帰写像) の実行: tの状態を履歴に保存し、t+1の_next状態を計算 (CSC前のベース) ---
        self._save_history(E_env_t, A_t_initial)
        self._update_layers(E_env_t, A_t_initial)

        # --- 4. 意識的安定化条件 (CSC) の実行: 努力 K の計測 ---
        A_final_refined, K = self._run_csc_stabilization(E_env_t, A_t_initial)

        # --- 5. 状態のコミット: 安定化された_next状態をcurrent状態に ---
        self._commit_state()

        # --- 6. 安定化された最終V(t)の計算 (PTRE effort K を適用) ---
        V_t, V_terms = self._calculate_V_from_state(K, state_prefix='')

        # --- 7. スキル学習則 A-TDL の実行 (TD誤差の計算とW_Cの更新) ---
        TD_error = 0.0
        tdl_norm = 0.0
        # T_BPTT分の履歴が溜まったら学習開始
        if len(self.learner.history) >= cfg.T_BPTT and self.t > 1:
            # TD誤差の計算: R(t-1) + gamma * V(t) - V(t-1)
            TD_error = self.learner.learn_step(V_t, self.V_prev, self.learner.history[-2], self.theta)

            # W_Cの更新
            tdl_norm = self._apply_tdl_gradients(TD_error, self.learner.history[-2])

        self.V_prev = V_t # V(t)を次のステップのV(t+1)予測のために保存

        # --- 8. 人格パラメータ $\theta$ の進化 (TD誤差の結果を用いて更新) ---
        # TDLが実行された場合にのみ$\theta$を更新
        delta_theta_norm = 0.0
        if self.t > cfg.T_BPTT:
            delta_theta_norm = self._evolve_theta(TD_error, K)

        # --- 9. 出力 ---
        Var_ctrl = V_terms['Var_ctrl']
        tau_ctrl = 0.05
        is_stable = Var_ctrl <= tau_ctrl

        output = {
            'action': A_final_refined,
            'V_total': V_t,
            'V_base': V_terms['V_base'],
            'Effort_Cost': V_terms['Effort_Cost'],
            'K_effort': K, # 新しい出力: 努力 K (反復回数)
            'happiness_core': np.mean(self.H_pz),
            'uncertainty_core': np.mean(self.U_pz),
            'control_load': Var_ctrl,
            'is_stable': is_stable,
            'theta_snapshot': self.theta,
            'TD_error': TD_error,
            'Delta_Theta_Norm': delta_theta_norm, # 人格進化の変動ノルム
            'TDL_W_norm': tdl_norm # スキル学習の変動ノルム
        }

        return output


# --- 実行例 ---

if __name__ == '__main__':
    print("--- Alice Architecture V3.0 - 最終統合コア (PTRE F_total 完全統合) ---")
    alice = AliceArchitecture()
    print(f"知性コア次元数 (C): {alice.config.N_C}, BPTT窓 (T_BPTT): {alice.config.T_BPTT}")
    print(f"初期 $\\theta^\\kappa$ (努力コスト感応度): {alice.theta['theta_kappa']:.6f}, $\\theta^\\beta$ (努力コスト指数): {alice.theta['theta_beta']:.2f}")

    # 状態のトラッキング
    time_steps = 30 # T_BPTT=16を超えて学習が始まるように設定
    # 報酬と入力を設定し、不安定な状況をシミュレート
    # 安定 -> 不安定（低報酬） -> 回復（高報酬）のシーケンス
    rewards = [0.1] * 5 + [-0.9] * 5 + [0.8] * 10 + [0.1] * 10
    inputs = ["observe environment"] * time_steps

    print("\n--- Alice Architecture シミュレーション開始 ---")

    initial_W_C_norm = np.linalg.norm(alice.W_C)
    initial_theta_kappa = alice.theta['theta_kappa']

    print(f"[t | R] V_total | V_base | Effort K | H | U | $\\theta^\\kappa$ | TD Error | TDL Norm")
    print("-" * 90)

    for i in range(time_steps):
        user_input = inputs[i]
        reward = rewards[i]

        result = alice.step(user_input, reward)

        # action_symbol = '🟢' if result['action'] > 0 else ('🔴' if result['action'] < 0 else '🟡')

        learning_status = f"{result['TDL_W_norm']:.2e}" if result['TDL_W_norm'] > 0 else "---"

        print(
            f"[{alice.t:02d} | {reward: 4.1f}] "
            f"V={result['V_total']: 6.2f} "
            f"({result['V_base']: 5.2f} - {result['Effort_Cost']: 4.2f}) | "
            f"K={result['K_effort']: 02d} | "
            f"H={result['happiness_core']: 4.2f} U={result['uncertainty_core']: 4.2f} | "
            f"$\\theta^\\kappa$={result['theta_snapshot']['theta_kappa']: 5.3f} | "
            f"TD={result['TD_error']: 5.2f} | "
            f"TDL={learning_status}"
        )

    final_W_C_norm = np.linalg.norm(alice.W_C)
    final_theta_kappa = alice.theta['theta_kappa']

    print("\n--- 最終学習と進化のチェック ---")
    print(f"初期 W_C ノルム: {initial_W_C_norm:.6f}")
    print(f"最終 W_C ノルム: {final_W_C_norm:.6f} ({'スキル学習発生' if abs(final_W_C_norm - initial_W_C_norm) > 1e-7 else '学習未発生'})")
    print(f"初期 $\\theta^\\kappa$: {initial_theta_kappa:.6f}")
    print(f"最終 $\\theta^\\kappa$: {final_theta_kappa:.6f} ({'人格進化発生' if abs(final_theta_kappa - initial_theta_kappa) > 1e-7 else '進化未発生'})")
    print(f"最終 $\\theta^{\\gamma_K}$: {alice.theta['theta_gamma_K']:.6f}")
