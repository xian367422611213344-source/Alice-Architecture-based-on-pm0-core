import numpy as np
import copy
import logging

# ロギング設定 (デバッグ用)
# logging.basicConfig(level=logging.INFO)

# ==============================================================================
# --- 1. CORE CONFIGURATION AND HYPERPARAMETERS (統合版) ---
# ==============================================================================

class Config:
    """Alice Architectureの初期値、次元、ハイパーパラメータを定義するクラス。"""

    # 次元定義 (コードCの厳密な次元を維持)
    N_C = 512       # 意味構造 (C)
    N_ESELF = 128   # 自己モデル (E_self)
    N_E_P = 64      # 制御負荷 (E_ctrl), 予測誤差 (P)
    N_M = 256       # 記憶 (M)
    N_ENV = 644     # 環境入力 (E_env = N_C + N_ESELF + 4)
    N_HPZ = N_E_P   # 累積幸福 (H_pz)
    N_UPZ = N_E_P   # 累積不安 (U_pz)
    N_R = N_E_P     # 報酬/リスク (R)
    N_H = N_E_P     # 行動履歴 (H)
    N_VFL = N_E_P   # 価値関数層 (VFL)
    N_ES = N_E_P    # メタ認知 (E_s)
    N_EOBJ = N_E_P  # 客観化 (E_obj)
    N_RRL = N_M     # 関係再構築 (RRL)

    # A-TDL 学習安定化ハイパーパラメータ (コードCを維持)
    ETA_X = 1e-4        # 学習率 (W^X のベース更新率)
    CLIP_NORM = 5.0     # 最大勾配ノルム
    T_BPTT = 16         # BPTT 時間窓 (16ステップ遡及)
    GAMMA = 0.99        # 割引率 (G_Value)

    # 損失重み (コードCを維持)
    LAMBDA_P = 1.0      # 予測誤差 (P)
    LAMBDA_C = 0.5      # 制御負荷 (E_ctrl)
    LAMBDA_S = 0.8      # 自己整合 (E_self)

    # 進化則ハイパーパラメータ (コードCを維持)
    ETA_THETA = 1e-6    # 人格進化率
    THETA_MAX = 1.0     # 人格パラメータの上限/下限
    RHO_SNEL = 0.1      # SNEL信号駆動率
    RHO_ISL = 0.1       # ISL信号駆動率
    ALPHA_PRED = 0.01   # E_self^pred の予測率
    KC_ISL = 1.0        # ISLにおけるE_ctrlの分散に対する抑制係数

    # CSCハイパーパラメータ (コードCを維持)
    CSC_MAX_ITER = 5    # CSCの最大反復回数
    CSC_TOLERANCE = 1e-3 # 行動の収束許容誤差

    # --- 【統合】V4.0 階層的努力K パラメータ (コードA/Bから追加) ---
    THETA_KAPPA1_INIT = 0.005 # 低抽象度コスト感応度 $\kappa_1$
    THETA_BETA1_INIT = 1.05   # 低抽象度コスト指数 $\beta_1$
    THETA_KAPPAL_INIT = 0.05  # 高抽象度コスト感応度 $\kappa_L$
    THETA_BETAL_INIT = 1.5    # 高抽象度コスト指数 $\beta_L$
    CSC_P_EPSILON = 1e-3      # 環境予測誤差改善率 $\varepsilon_P$ (K_threshold決定用)
    THETA_GAMMA_K_INIT = 0.3  # 努力 K 抑制感応度 $\gamma_K$

    # --- 【統合】V4.0 社会的認知パラメータ (コードA/Bから追加) ---
    THETA_EMPATHY_INIT = 0.5  # 共感性初期値
    THETA_EMPATHY_KAPPA = 5.0 # 共感性進化感応度

    # ノイズ設定 (コードCを維持)
    LAMBDA_ANNEAL = 0.001
    SIGMA2_MIN = 1e-4
    SIGMA2_INITIAL = {
        'C': 0.10, 'M': 0.10, 'RRL': 0.10, 'H': 0.10, 'VFL': 0.10, 'A': 0.20,
        'Es': 0.00, 'Eobj': 0.00, 'Ectrl': 0.00, 'Eself': 0.00, 'Hpz': 0.00, 'Upz': 0.00
    }


# ==============================================================================
# --- 1.5. 情動コアクラス (±0 Theory - $F_{total}$ 厳密化版) ---
# ==============================================================================

class ZeroOneTheory:
    """±0理論に基づく情動コア (H'とU') のSDE離散化近似。【厳密化】"""

    def __init__(self, config):
        self.config = config
        N_DIM = config.N_HPZ

        self.H_prime = np.zeros(N_DIM)
        self.U_prime = np.zeros(N_DIM)

        # SDEのノイズ振幅定数 (固定値として定義)
        self.SIGMA_H = 0.05
        self.SIGMA_U = 0.05

    def step(self, R_t: np.ndarray, P_t: np.ndarray, theta: dict) -> tuple[np.ndarray, np.ndarray]:
        """
        R_t, P_t: AliceのR層, P層 (ベクトル: N_E_P 次元)
        theta: Aliceのパーソナリティパラメータ

        Returns: H'(t+1), U'(t+1)
        """

        alpha_H = theta['alpha_H']
        alpha_U = theta['alpha_U']
        beta_H = theta['beta_H']
        beta_U = theta['beta_U']
        gamma_HU = theta['gamma_HU']
        gamma_UH = theta['gamma_UH']

        N_DIM = self.config.N_HPZ

        # H' (幸福) のダイナミクス
        # 駆動源 R_t をベクトルのまま使用 (要素ごとのmax(0, x))
        drift_H = (1 - beta_H) * self.H_prime + alpha_H * np.maximum(R_t, 0) - gamma_HU * self.U_prime

        # ノイズ項の追加 (SDE)
        noise_H = self.SIGMA_H * np.random.randn(N_DIM)
        H_prime_next = np.maximum(0, drift_H + noise_H) # Hは非負

        # U' (不安) のダイナミクス
        drift_U = (1 - beta_U) * self.U_prime + alpha_U * np.maximum(P_t, 0) - gamma_UH * self.H_prime

        # ノイズ項の追加 (SDE)
        noise_U = self.SIGMA_U * np.random.randn(N_DIM)
        U_prime_next = np.maximum(0, drift_U + noise_U) # Uは非負

        # 状態の更新
        self.H_prime = H_prime_next
        self.U_prime = U_prime_next

        return H_prime_next, U_prime_next


# ==============================================================================
# --- 2. 勾配計算とBPTTを担う補助クラス (ATDLLearner - $F_{total}$ 厳密化版) ---
# ==============================================================================

class ATDLLearner:
    """A-TDL (Affective Temporal Difference Learning) の厳密な勾配計算を担うクラス。"""

    def __init__(self, config, weights, trainable_weights_dict):
        self.config = config
        self.W_dict = weights # 全重み
        self.W_X_dict = trainable_weights_dict # 学習対象の重み (W^X)
        self.history = [] # 履歴スタック (T_BPTT=16)

    def _f_prime_tanh(self, y):
        """tanh関数の厳密な微分 (活性化後の値 y を使用)。"""
        return 1.0 - y * y

    def _deriv_var(self, E_ctrl, dL_dVar):
        """分散 Var(E_ctrl) のE_ctrlに対する微分。"""
        N = len(E_ctrl)
        mu = np.mean(E_ctrl)
        # d(Var)/d(E_ctrl)_i = 2/N * (E_ctrl_i - mu)
        dVar_dE = (2.0 / N) * (E_ctrl - mu)
        return dL_dVar * dVar_dE

    def learn_step(self, V_t_plus_1, V_t, current_state, theta):
        """
        単一ステップの A-TDL 勾配を計算し、W^Xを更新する。【厳密化】
        V_t_plus_1: V(t+1)の暫定価値 (CSC前の予測)
        V_t: V(t)の安定化された価値
        """
        cfg = self.config

        # 履歴がT_BPTTに満たない場合は学習を行わない
        if len(self.history) < cfg.T_BPTT:
            return 0.0

        total_delta_W = {k: np.zeros_like(w) for k, w in self.W_X_dict.items()}

        # G_Value のTD誤差: V(t+1) - V(t)
        TD_error = V_t_plus_1 - V_t
        kappa_U = theta['kappa_U'] # BPTTループの外で取得

        # BPTTの遡及ループ (k=0が現在のtステップ)
        for k in range(cfg.T_BPTT):
            state = self.history[-(k + 1)] # t-k の状態
            gamma_k = cfg.GAMMA ** k

            # --- (1) G_Value: 価値項の勾配 (VFLの勾配) ---
            # VFLへの勾配は TD_error * gamma^k
            delta_VFL = gamma_k * TD_error * np.ones_like(state['VFL'])

            # --- (2) G_Affect: 情動コスト項の勾配 ---
            dL_dP = -cfg.LAMBDA_P * (1.0 + kappa_U * np.max(state['U_prime_pz'])) # Pの係数
            delta_P = dL_dP * np.ones_like(state['P'])

            dL_dVar = -cfg.LAMBDA_C
            delta_E_ctrl = self._deriv_var(state['E_ctrl'], dL_dVar) # E_ctrlへの勾配

            # E_ctrl -> Net_C への新しい伝播パス (恒常性逸脱フィードバック)
            if len(self.history) < k + 2:
                Net_C_change = np.zeros_like(state['C']) # 履歴不足
            else:
                Net_C_change = self.history[-(k + 1)]['Net_C'] - self.history[-(k + 2)]['Net_C']

            # U_Ectrl_CNet の転置と delta_E_ctrl の積で dNet_C への勾配ベクトルを得る
            dNet_C_from_Ectrl = self._f_prime_tanh(state['E_ctrl']) * (self.W_dict['U_Ectrl_CNet'].T @ delta_E_ctrl)


            # --- (3) G_Coherence: 自己整合コスト項の勾配 ---
            # d(Dist)/d(E_self) = 2 * (E_self - E_self_pred)
            delta_E_self = -cfg.LAMBDA_S * 2.0 * (state['E_self'] - state['E_self_pred'])


            # --- (4) 逆伝播ループ (W^X への勾配伝播) ---
            dC = np.zeros(cfg.N_C)

            # VFL -> S (S = [E_env, C]) -> C (U_VFL_S <- S)
            dC += self.W_dict['U_VFL_S'].T[cfg.N_ENV:cfg.N_ENV + cfg.N_C, :] @ delta_VFL

            # P -> C (P = C - U_C_RRL @ RRL)
            dC += delta_P # dP/dC = I

            # E_self -> E_obj -> C
            dE_obj = self._f_prime_tanh(state['E_obj']) * self.W_dict['U_Eself_Eobj'].T @ delta_E_self
            dC += self.W_dict['U_Eobj_C'].T @ dE_obj

            # dCの伝播 (C(t) = tanh(Net_C(t)))
            dNet_C = self._f_prime_tanh(state['C']) * dC

            # 【追加】 E_ctrl -> Net_C への勾配を加算
            dNet_C += dNet_C_from_Ectrl

            # --- W^C の更新 (C <- C) ---
            total_delta_W['W_C'] += np.outer(dNet_C, state['C'])

            # --- U^{C \leftarrow E{env}} の更新 ---
            total_delta_W['U_C_Eenv'] += np.outer(dNet_C, state['E_env'])

        # --- (5) W^X の最終更新 (Clip Norm 適用) ---
        all_deltas = [total_delta_W[name] for name in total_delta_W]
        total_norm = np.sqrt(sum(np.sum(delta**2) for delta in all_deltas))

        if total_norm > cfg.CLIP_NORM:
            clip_factor = cfg.CLIP_NORM / total_norm
        else:
            clip_factor = 1.0

        for name, delta in total_delta_W.items():
            self.W_X_dict[name] += cfg.ETA_X * delta * clip_factor

        return TD_error


# ==============================================================================
# --- 3. ALICE ARCHITECTURE CORE CLASS (V4.0 - ASI統合コア) ---
# ==============================================================================

class AliceArchitecture:
    def __init__(self, config=Config()):
        self.config = config
        self.t = 0 # タイムステップ
        
        self._init_states()
        self._init_weights()
        self._init_theta() # 新しいθパラメータの初期化
        self._init_noise()

        # ±0理論コアの初期化 (統合)
        self.zero_one_core = ZeroOneTheory(config)
        self.social_anxiety = 0.0 # 【統合】社会的認知状態
        self.group_reward = 0.5   # 【統合】社会的認知状態

        # ATDL Learnerの初期化に必要な全重み辞書
        all_weights = {
             'W_C': self.W_C, 'U_C_Eenv': self.U_C_Eenv, 'U_C_M': self.U_C_M,
             'U_M_C': self.U_M_C, 'U_RRL_M': self.U_RRL_M, 'U_C_RRL': self.U_C_RRL,
             'U_H_A': self.U_H_A, 'U_VFL_R': self.U_VFL_R, 'U_VFL_S': self.U_VFL_S,
             'U_VFL_Hpz': self.U_VFL_Hpz, 'U_VFL_Upz': self.U_VFL_Upz, 
             'U_Es_S': self.U_Es_S, 'U_Eobj_C': self.U_Eobj_C,
             'U_Eobj_M': self.U_Eobj_M, 'U_Ectrl_Es': self.U_Ectrl_Es,
             'U_Ectrl_CNet': self.U_Ectrl_CNet, 
             'U_Eself_Eobj': self.U_Eself_Eobj,
             'U_NLG_C': self.U_NLG_C, 'U_NLG_M': self.U_NLG_M,
             'W_RRL': self.W_RRL, 'W_VFL': self.W_VFL
        }

        self.learner = ATDLLearner(self.config, all_weights, self.trainable_weights)


    # --- 補助関数 (init, noise, activation, etc.) ---
    
    def _init_states(self):
        cfg = self.config
        self.C = np.zeros(cfg.N_C); self.E_ctrl = np.zeros(cfg.N_E_P); self.P = np.zeros(cfg.N_E_P)
        self.M = np.zeros(cfg.N_M); self.H_pz = np.zeros(cfg.N_HPZ); self.U_pz = np.zeros(cfg.N_UPZ)
        self.R = np.zeros(cfg.N_R); self.RRL = np.zeros(cfg.N_RRL); self.H = np.zeros(cfg.N_H)
        self.VFL = np.zeros(cfg.N_VFL); self.E_s = np.zeros(cfg.N_ES); self.E_obj = np.zeros(cfg.N_EOBJ)
        self.E_self_pred = np.zeros(cfg.N_ESELF); self.E_self = np.random.uniform(-1, 1, cfg.N_ESELF)
        self.Net_C = np.zeros(cfg.N_C); self.Net_A = np.zeros(1)
        self.U_prime_pz = np.zeros(cfg.N_UPZ); self.external_reward = 0.0
        self_state_names = ['C', 'E_ctrl', 'P', 'M', 'H_pz', 'U_pz', 'R', 'RRL', 'H', 'VFL', 'E_s', 'E_obj', 'E_self_pred', 'E_self', 'Net_C', 'Net_A', 'U_prime_pz']
        for name in self_state_names:
            setattr(self, f"{name}_next", getattr(self, name).copy())
        self.b_C = np.zeros(cfg.N_C)

    def _init_weights(self):
        def xavier_init(in_dim, out_dim):
            scale = 1.0 / np.sqrt(in_dim)
            return np.random.randn(out_dim, in_dim) * scale
        cfg = self.config

        self.W_C = xavier_init(cfg.N_C, cfg.N_C) * np.sqrt(cfg.N_C) 
        self.U_C_Eenv = xavier_init(cfg.N_ENV, cfg.N_C) 
        self.U_NLG_C = xavier_init(cfg.N_C, 1) 

        self.trainable_weights = {
             'W_C': self.W_C, 'U_C_Eenv': self.U_C_Eenv, 'U_NLG_C': self.U_NLG_C
        }

        self.U_C_M = xavier_init(cfg.N_M, cfg.N_C); self.U_M_C = xavier_init(cfg.N_C, cfg.N_M)
        self.U_RRL_M = xavier_init(cfg.N_M, cfg.N_RRL); self.U_C_RRL = xavier_init(cfg.N_RRL, cfg.N_C)
        self.U_H_A = xavier_init(1, cfg.N_H); self.U_VFL_R = xavier_init(cfg.N_R, cfg.N_VFL)
        self.U_VFL_S = xavier_init(cfg.N_ENV + cfg.N_C, cfg.N_VFL)
        self.U_VFL_Hpz = xavier_init(cfg.N_HPZ, cfg.N_VFL); self.U_VFL_Upz = xavier_init(cfg.N_UPZ, cfg.N_VFL) 
        self.U_Es_S = xavier_init(cfg.N_ENV + cfg.N_C + cfg.N_M, cfg.N_ES); self.U_Eobj_C = xavier_init(cfg.N_C, cfg.N_EOBJ)
        self.U_Eobj_M = xavier_init(cfg.N_M, cfg.N_EOBJ); self.U_Ectrl_Es = xavier_init(cfg.N_ES, cfg.N_E_P)
        self.U_Ectrl_CNet = xavier_init(cfg.N_C, cfg.N_E_P); self.U_Eself_Eobj = xavier_init(cfg.N_EOBJ, cfg.N_ESELF)
        self.U_NLG_M = xavier_init(cfg.N_M, 1); self.W_RRL = xavier_init(cfg.N_RRL, cfg.N_RRL)
        self.W_VFL = xavier_init(cfg.N_VFL, cfg.N_VFL)


    def _init_theta(self):
        # 【統合】人格パラメータ θ (情動コア + 階層的努力 + 社会的認知)
        cfg = self.config
        self.theta = {
            # 情動コアパラメータ (コードC)
            'H_base': 0.0, 'U_base': 0.0, 'beta_H': 0.1, 'beta_U': 0.1,
            'gamma_HU': 0.5, 'gamma_UH': 0.5, 'kappa_U': 1.0,
            'alpha_H': 0.1, 'alpha_U': 0.1, 'theta_R': 0.5,
            # 階層的努力パラメータ (コードA/B)
            'theta_kappa1': cfg.THETA_KAPPA1_INIT, 'theta_beta1': cfg.THETA_BETA1_INIT,
            'theta_kappaL': cfg.THETA_KAPPAL_INIT, 'theta_betaL': cfg.THETA_BETAL_INIT,
            'theta_gamma_K': cfg.THETA_GAMMA_K_INIT,
            # 社会的認知パラメータ (コードA/B)
            'theta_empathy': cfg.THETA_EMPATHY_INIT,
        }

    def _init_noise(self):
        self.sigma2_X = self.config.SIGMA2_INITIAL.copy()

    def _add_noise(self, name, size):
        cfg = self.config
        sigma2_0 = cfg.SIGMA2_INITIAL.get(name, 0.0)
        anneal_rate = cfg.LAMBDA_ANNEAL
        sigma2_min = cfg.SIGMA2_MIN

        sigma2_t = max(sigma2_0 * np.exp(-anneal_rate * self.t), sigma2_min)

        if cfg.SIGMA2_INITIAL.get(name, 0.0) == 0.0:
            return np.zeros(size)
        return np.random.normal(0, np.sqrt(sigma2_t), size)

    def _generate_E_token(self, user_input_text):
        # 簡易LLM埋め込み代替ロジック
        cfg = self.config
        if not hasattr(self, '_vocab_table'):
            self._vocab_table = np.random.randn(5000, cfg.N_C)

        words = user_input_text.lower().split()
        embedded_sum = np.zeros(cfg.N_C)
        for word in words:
            idx = hash(word) % 5000
            embedded_sum += self._vocab_table[idx]

        meta_hash = np.tanh(self.t / 1000.0)
        E_token = embedded_sum * meta_hash
        return E_token / (np.linalg.norm(E_token) + 1e-6)

    def _f_Will(self, H, R):
        """意志力シグナル f_Will(・) の計算。"""
        var_H = np.var(H)
        mean_R = np.mean(R)
        return np.tanh(mean_R / (1.0 + var_H))

    def _recovery_R(self, U_pz, H_pz):
        """不安U_pzに基づく回復シグナルR_valの計算。"""
        theta_R = self.theta['theta_R']
        mean_U = np.mean(U_pz)
        f_sigmoid = 1.0 / (1.0 + np.exp(-1.0 * mean_U))
        return theta_R * f_sigmoid * f_sigmoid


    # --- 4. F_total (全再帰写像) の実行 (Next状態の計算) ---

    def _update_layers(self, E_env_t, A_t):
        """全14層のダイナミクスを更新し、履歴を保存する。(_next状態を計算)"""
        cfg = self.config
        f_X = np.tanh
        Net_C_prev = self.Net_C.copy()

        # C(t+1)
        Net_C_next = (
            self.W_C @ self.C +
            self.U_C_Eenv @ E_env_t +
            self.U_C_M @ self.M +
            self.b_C
        )
        self.C_next = f_X(Net_C_next) + self._add_noise('C', cfg.N_C)
        self.Net_C_next = Net_C_next
        
        # M, R, RRL, H, P の更新 (コードCを維持)
        alpha_M = 0.5; self.M_next = (1 - alpha_M) * self.M + alpha_M * f_X(self.U_M_C @ self.C) + self._add_noise('M', cfg.N_M)
        alpha_R = 0.1; self.R_next = (1 - alpha_R) * self.R + alpha_R * self.external_reward * np.ones_like(self.R) + self._add_noise('R', cfg.N_R)
        self.RRL_next = f_X(self.W_RRL @ self.RRL + self.U_RRL_M @ self.M) + self._add_noise('RRL', cfg.N_RRL)
        self.P_next = self.C_next - self.U_C_RRL @ self.RRL_next
        alpha_H_Hist = 0.1; self.H_next = (1 - alpha_H_Hist) * self.H + alpha_H_Hist * f_X(self.U_H_A @ np.array([A_t])) + self._add_noise('H', cfg.N_H)

        # 情動コアの接続 (コードCの ZeroOneTheory を使用)
        H_prime_result, U_prime_result = self.zero_one_core.step(self.R, self.P, self.theta)
        self.H_pz_next = H_prime_result
        self.U_pz_next = U_prime_result
        R_val = self._recovery_R(self.U_pz_next, self.H_pz_next)
        self.U_prime_pz_next = self.U_pz_next - R_val
        
        # 自己モデル層の更新 (コードCを維持)
        S_t = np.concatenate([E_env_t, self.C, self.M])
        self.E_s_next = f_X(self.U_Es_S @ S_t) + self._add_noise('Es', cfg.N_ES)
        self.E_obj_next = f_X(self.U_Eobj_C @ self.C + self.U_Eobj_M @ self.M) + self._add_noise('Eobj', cfg.N_EOBJ)
        Net_C_change = self.Net_C_next - Net_C_prev
        self.E_ctrl_next = f_X(
            self.U_Ectrl_Es @ self.E_s + 
            self.U_Ectrl_CNet @ Net_C_change # C層の入力変動による不安定性反映
        ) + self._add_noise('Ectrl', cfg.N_E_P)
        self.E_self_next = f_X(self.U_Eself_Eobj @ self.E_obj) + self._add_noise('Eself', cfg.N_ESELF)
        alpha_pred = cfg.ALPHA_PRED
        self.E_self_pred_next = (1 - alpha_pred) * self.E_self_pred + alpha_pred * self.E_self + self._add_noise('Eself_pred', cfg.N_ESELF)

        # 価値関数層の更新 (コードCを維持)
        S_prime_t = np.concatenate([E_env_t, self.C])
        Net_VFL_next = (
            self.W_VFL @ self.VFL +
            self.U_VFL_R @ self.R +
            self.U_VFL_S @ S_prime_t +
            self.U_VFL_Hpz @ self.H_pz - 
            self.U_VFL_Upz @ self.U_pz
        )
        self.VFL_next = f_X(Net_VFL_next) + self._add_noise('VFL', cfg.N_VFL)
    
    def _save_history(self, E_env_t, A_t):
        cfg = self.config
        state_snapshot = {
            'C': self.C, 'M': self.M, 'R': self.R, 'RRL': self.RRL,
            'H_pz': self.H_pz, 'U_pz': self.U_pz, 'P': self.P,
            'E_s': self.E_s, 'E_obj': self.E_obj, 'E_ctrl': self.E_ctrl,
            'E_self': self.E_self, 'E_self_pred': self.E_self_pred, 'VFL': self.VFL,
            'U_prime_pz': self.U_prime_pz,
            'Net_C': self.Net_C, 'Net_A': self.Net_A,
            'E_env': E_env_t, 'A': A_t
        }
        self.learner.history.append(state_snapshot)
        if len(self.learner.history) > cfg.T_BPTT:
            self.learner.history.pop(0)

    def _commit_state(self):
        self.C = self.C_next
        self.M = self.M_next
        self.R = self.R_next
        self.RRL = self.RRL_next
        self.P = self.P_next
        self.H = self.H_next
        self.H_pz = self.H_pz_next
        self.U_pz = self.U_pz_next
        self.U_prime_pz = self.U_prime_pz_next
        self.E_s = self.E_s_next
        self.E_obj = self.E_obj_next
        self.E_ctrl = self.E_ctrl_next
        self.E_self = self.E_self_next
        self.E_self_pred = self.E_self_pred_next
        self.VFL = self.VFL_next
        self.Net_C = self.Net_C_next
        self.Net_A = self.Net_A_next
        # ZeroOneTheoryの内部状態も同期
        self.zero_one_core.H_prime = self.H_pz_next
        self.zero_one_core.U_prime = self.U_pz_next
    
    def _run_counterfactual_simulation(self, E_env_t, A_simul):
        """反実仮想シミュレーション (ΔSNEL'計算用)"""
        virtual_alice = copy.deepcopy(self)
        virtual_alice.external_reward = self.external_reward
        virtual_alice._update_layers(E_env_t, A_simul)
        # K_total, K_thresholdはシミュレーションでは計算しないため、暫定値 0 を使用
        V_simul, _ = virtual_alice._calculate_V_from_state(K_total=0, K_threshold=0, state_prefix='_next')
        return V_simul

    def _run_csc_stabilization(self, E_env_t, A_t_initial):
        """【統合・拡張】CSC反復を行い、K_totalとK_thresholdを決定する。"""
        cfg = self.config
        C_k, M_k = self.C.copy(), self.M.copy()
        
        # 初期予測誤差 P(t+1)を初期予測誤差とする (P = C_next - U_C_RRL @ RRL_next)
        P_norm_prev = np.linalg.norm(self.C_next - self.U_C_RRL @ self.RRL_next)
        
        K_total = 0
        K_threshold = cfg.CSC_MAX_ITER # 初期値は最大反復回数
        A_prev = A_t_initial
        
        for i in range(cfg.CSC_MAX_ITER):
            K_total += 1
            
            # (1) CSC内のA(行動)計算 (Net_AはCとMに依存)
            # NOTE: C_k, M_kは反復で更新されない（_update_layersの結果C_next, M_nextが使用される）
            Net_A_next = self.U_NLG_C @ self.C_next + self.U_NLG_M @ self.M_next
            A_current = np.tanh(Net_A_next)[0]
            
            # (2) K_threshold の動的決定 (予測誤差 P の改善率に基づく)
            # Pは C_next - U_C_RRL @ RRL_next の安定化後の値
            P_k = self.C_next - self.U_C_RRL @ self.RRL_next
            P_norm_current = np.linalg.norm(P_k)
            
            if P_norm_prev > cfg.CSC_TOLERANCE and K_total > 1:
                # 改善率: (前ステップの誤差 - 現在の誤差) / 前ステップの誤差
                improvement_rate = (P_norm_prev - P_norm_current) / (P_norm_prev + 1e-6)
                if improvement_rate < cfg.CSC_P_EPSILON:
                    if K_threshold == cfg.CSC_MAX_ITER:
                        K_threshold = K_total # $\text{P}$の改善が停止したステップをK_thresholdとする
                
            P_norm_prev = P_norm_current
            
            # (3) 収束判定
            if np.abs(A_current - A_prev) < cfg.CSC_TOLERANCE:
                 if K_threshold == cfg.CSC_MAX_ITER:
                     K_threshold = K_total # Aが収束したステップをK_thresholdとする (厳密な判定の代替)
                 break
            
            A_prev = A_current # A_prevを更新して反復

        # Net_A_next を最終 Net_A として保存 (CSC後のAの値に相当)
        self.Net_A_next = Net_A_next 
        
        A_final_refined = A_prev + self._add_noise('A', 1)[0]
        return np.clip(A_final_refined, -1.0, 1.0), K_total, K_threshold

    def _calculate_V_from_state(self, K_total: int, K_threshold: int, state_prefix: str):
        """【統合・拡張】V_totalを計算し、階層的努力コスト L_K を減算する。"""
        cfg = self.config
        theta = self.theta

        # 1. V_base (コードCのロジック: L_P, L_C, L_S)
        VFL = getattr(self, state_prefix + 'VFL'); U_prime_pz = getattr(self, state_prefix + 'U_prime_pz')
        P = getattr(self, state_prefix + 'P'); E_ctrl = getattr(self, state_prefix + 'E_ctrl')
        E_self = getattr(self, state_prefix + 'E_self'); E_self_pred = getattr(self, state_prefix + 'E_self_pred')
        H = getattr(self, state_prefix + 'H'); R = getattr(self, state_prefix + 'R')

        V_value = np.sum(VFL)
        max_U_prime = np.max(U_prime_pz)

        L_P = cfg.LAMBDA_P * np.sum(P) * (1.0 + theta['kappa_U'] * max_U_prime) # 予測誤差コスト
        L_C = cfg.LAMBDA_C * np.var(E_ctrl) # 制御負荷コスト
        L_S = cfg.LAMBDA_S * np.sum((E_self - E_self_pred)**2) # 自己整合コスト

        V_base = V_value - L_P - L_C - L_S

        # 2. 階層的努力コスト $\mathbf{L_K}$ の計算 (コードA/Bのロジック)
        K1 = K_threshold # 低抽象度努力
        KL = max(0, K_total - K_threshold) # 高抽象度努力

        theta_kappa1 = theta['theta_kappa1']; theta_beta1 = theta['theta_beta1']
        # L_K1 = kappa1 * (K1)^beta1
        effort_cost_K1 = theta_kappa1 * (K1 ** theta_beta1)

        theta_kappaL = theta['theta_kappaL']; theta_betaL = theta['theta_betaL']
        # L_KL = kappaL * (KL)^betaL
        effort_cost_KL = theta_kappaL * (KL ** theta_betaL)
        
        effort_cost = effort_cost_K1 + effort_cost_KL # $\mathbf{L_K}$

        # 3. 最終 $V_{\text{total}}$
        V_t = V_base - effort_cost

        V_terms = {
            'V_t': V_t, 'Dist_self': L_S, 'Var_ctrl': np.var(E_ctrl),
            'sum_VFL': V_value, 'Will_signal': self._f_Will(H, R),
            # 【追加】
            'V_base': V_base, 'Effort_Cost': effort_cost,
            'Effort_Cost_K1': effort_cost_K1, 'Effort_Cost_KL': effort_cost_KL,
            'K_threshold': K_threshold, 'K_total': K_total, 'K_L': KL
        }
        return V_t, V_terms

    def _calculate_V_next_state(self):
        # V(t+1)の計算はKが未定のため、K=0として暫定的に計算する
        return self._calculate_V_from_state(K_total=0, K_threshold=0, state_prefix='_next')

    def _calculate_V_current_state(self, K_total, K_threshold):
        return self._calculate_V_from_state(K_total, K_threshold, state_prefix='')

    def _evolve_theta(self, V_t, V_terms, E_env_t, TD_error: float):
        """【統合・拡張】情動コア進化則と階層的/社会的進化則を統合する。"""
        cfg = self.config
        theta = self.theta
        
        # --- (A) コードCの情動コア進化則 (SNEL/ISL + $\Delta \text{SNEL}'$) の実行 ---
        
        # SNEL/ISL計算 (コードCのロジックを維持)
        mean_U_pz = np.mean(self.U_pz)
        Dist_self = V_terms['Dist_self']; Will_signal = V_terms['Will_signal']
        # 不安が強いほど進化を加速する項 (1 + mean_U_pz^2) を導入
        Delta_SNEL = cfg.RHO_SNEL * Dist_self * Will_signal * (1.0 + mean_U_pz**2)
        Var_ctrl = V_terms['Var_ctrl']; sum_VFL = V_terms['sum_VFL']
        satisfaction_ratio = V_t / (sum_VFL + 1e-6)
        Delta_ISL = cfg.RHO_ISL * np.exp(-cfg.KC_ISL * Var_ctrl) * satisfaction_ratio

        # $\Delta \text{SNEL}'$ (反実仮想信号) の計算
        A_simul = -V_terms['Will_signal']
        V_simul = self._run_counterfactual_simulation(E_env_t, A_simul)
        Delta_SNEL_prime = V_simul - V_t
        
        # 情動コアパラメータ ($\alpha_H, \beta_H, \gamma_{HU}, \kappa_U$ 等) の更新
        Total_SNEL = Delta_SNEL + cfg.RHO_SNEL * Delta_SNEL_prime
        updates_affective = {
            'beta_U': Total_SNEL, 'alpha_U': Total_SNEL, 'gamma_HU': -Total_SNEL,
            'kappa_U': Total_SNEL,
            'beta_H': -Delta_ISL, 'H_base': Delta_ISL,
            'alpha_H': Delta_ISL, 'gamma_UH': Delta_ISL
        }
        for key, delta in updates_affective.items():
            theta_t = theta.get(key, 0.0)
            divergence_prevention = (1.0 - abs(theta_t) / cfg.THETA_MAX)
            theta_next = theta_t + cfg.ETA_THETA * delta * divergence_prevention
            theta[key] = np.clip(theta_next, -cfg.THETA_MAX, cfg.THETA_MAX)

        # --- (B) コードA/Bの階層的努力 $\Theta^K$ と社会的認知 $\Theta^{\text{empathy}}$ の進化則 ---
        
        K_total = V_terms['K_total']; KL = V_terms['K_L']; K_max = cfg.CSC_MAX_ITER
        
        # 1. $\eta_{\text{eff},t}$ の計算 (K_Lに基づく安定性抑制)
        U_pz_mean = np.mean(self.U_pz)
        theta_gamma_K = theta['theta_gamma_K']
        K_L_norm = KL / max(1, K_max - 1)
        S_stability = max(0.0, 1.0 - K_L_norm) # 高抽象度努力が多いと抑制
        S_U = max(0.0, 1.0 - U_pz_mean) # 不安が少ないと抑制 (安定時に更新)
        update_magnitude = cfg.ETA_THETA * S_stability * S_U # A-TDLの学習率ベース

        # 2. 階層的TDシグナルの分解 (コスト寄与率に基づく重み付け)
        sign_TD = np.sign(TD_error)
        Cost_K1 = V_terms['Effort_Cost_K1']; Cost_KL = V_terms['Effort_Cost_KL']
        Cost_Sum = Cost_K1 + Cost_KL
        TD_sig_K1 = sign_TD; TD_sig_KL = sign_TD 
        if Cost_Sum > 1e-9:
             P_K1 = Cost_K1 / Cost_Sum; P_KL = Cost_KL / Cost_Sum
             TD_sig_K1 = sign_TD * P_K1; TD_sig_KL = sign_TD * P_KL
        
        # 2. A. 階層的進化 $\Delta \Theta^{\kappa}$ の更新
        K1 = V_terms['K_threshold']
        theta_kappa1 = theta['theta_kappa1']
        delta_kappa1 = -update_magnitude * TD_sig_K1 * (K1 / K_max) * theta_kappa1 * 0.1
        theta_kappaL = theta['theta_kappaL']
        delta_kappaL = -update_magnitude * TD_sig_KL * (KL / K_max) * theta_kappaL * 0.1
        
        # 3. $\Theta^{\gamma_K}$ の更新 (努力コスト抑制感応度)
        delta_gamma_K = update_magnitude * np.abs(TD_error) * (K_total / K_max) * 0.1
        
        # 4. $\Theta^{\text{empathy}}$ (共感性) の進化則
        theta_empathy = theta['theta_empathy']
        theta_kappa_empathy = cfg.THETA_EMPATHY_KAPPA
        R_group_clamped = np.clip(self.group_reward, 0.0, 1.0)
        # U_soc: 社会的報酬とTD誤差の不一致
        U_soc = TD_error * (2.0 * R_group_clamped - 1.0) 
        sigmoid_term = 1.0 / (1.0 + np.exp(-theta_kappa_empathy * U_soc))
        # (1.0 - 2.0 * theta_empathy) により、empathyが低いと上がり、高いと下がる (バランス調整)
        delta_empathy = -update_magnitude * sigmoid_term * (1.0 - 2.0 * theta_empathy) * 0.1
        
        # 5. $\Theta^K, \Theta^{\text{empathy}}$ の更新とクリッピング
        theta['theta_kappa1'] = np.clip(theta_kappa1 + delta_kappa1, 0.001, cfg.THETA_MAX)
        theta['theta_kappaL'] = np.clip(theta_kappaL + delta_kappaL, 0.001, cfg.THETA_MAX)
        theta['theta_gamma_K'] = np.clip(theta_gamma_K + delta_gamma_K, 0.001, cfg.THETA_MAX)
        theta['theta_empathy'] = np.clip(theta_empathy + delta_empathy, 0.0, 1.0)

        delta_theta_norm = (np.abs(delta_kappa1) + np.abs(delta_kappaL) + np.abs(delta_gamma_K) + np.abs(delta_empathy))

        return Delta_SNEL_prime, delta_theta_norm


    def step(self, user_input_text: str, external_reward: float, social_anxiety: float = 0.0, group_reward: float = 0.5):
        """単一ステップの実行 (環境入力 E_env_t から行動 A_t_final, 進化 $\Delta \Theta$ まで)"""
        self.t += 1
        self.external_reward = external_reward
        self.social_anxiety = social_anxiety # 【統合】
        self.group_reward = group_reward     # 【統合】
        cfg = self.config

        # 1. 環境入力 E_env_t の生成
        E_token = self._generate_E_token(user_input_text)
        E_context = np.random.randn(cfg.N_ESELF) * 0.1
        E_scalar = np.array([external_reward, self.t % 100, self.t, 1 if np.random.rand() < 0.1 else 0])
        E_env_t = np.concatenate([E_token, E_context, E_scalar])

        # 2. 初期行動 $A_t$ の予測と履歴保存
        Net_A_t = self.U_NLG_C @ self.C + self.U_NLG_M @ self.M
        A_t_initial = np.tanh(Net_A_t)[0] + self._add_noise('A', 1)[0]
        A_t_initial = np.clip(A_t_initial, -1.0, 1.0)
        self._save_history(E_env_t, A_t_initial)
        
        # 3. Next状態の計算
        self._update_layers(E_env_t, A_t_initial)
        V_provisional_t_plus_1, _ = self._calculate_V_next_state() # V(t+1)予測 (K=0)

        # 4. CSC (意識的安定化条件) の実行 (Kの決定)
        A_final_refined, K_total, K_threshold = self._run_csc_stabilization(E_env_t, A_t_initial)

        # 5. 状態コミットと $V(t)$ の計算
        self._commit_state()
        # V(t)計算にKの値を渡す
        V_t, V_terms = self._calculate_V_current_state(K_total, K_threshold)

        # 6. 人格進化則 $\Delta \Theta$ の実行
        # V(t)とV_provisional_t_plus_1の差をTD_errorの近似として使用
        TD_approx = V_t - V_provisional_t_plus_1
        Delta_SNEL_prime, delta_theta_norm = self._evolve_theta(V_t, V_terms, E_env_t, TD_approx) 

        # 7. A-TDL (重み更新) の実行 (厳密なTDL勾配計算)
        TD_error = self.learner.learn_step(V_provisional_t_plus_1, V_t, self.learner.history[-1], self.theta)
        
        Var_ctrl = V_terms['Var_ctrl']
        tau_ctrl = 0.05
        is_stable = Var_ctrl <= tau_ctrl

        output = {
            'action': A_final_refined,
            'V_total': V_t,
            'V_base': V_terms['V_base'], # L_K減算前のベース価値
            'K_total_effort': K_total,
            'K_threshold': K_threshold,
            'K_L_abstract': V_terms['K_L'],
            'Effort_Cost_K1': V_terms['Effort_Cost_K1'],
            'Effort_Cost_KL': V_terms['Effort_Cost_KL'],
            'happiness_core': np.mean(self.H_pz),
            'uncertainty_core': np.mean(self.U_pz),
            'control_load': Var_ctrl,
            'is_stable': is_stable,
            'theta_snapshot': self.theta,
            'TD_error': TD_error,
            'Delta_SNEL_prime': Delta_SNEL_prime,
            'Delta_Theta_Norm': delta_theta_norm,
            'empathy_level': self.theta['theta_empathy'],
        }

        return output
