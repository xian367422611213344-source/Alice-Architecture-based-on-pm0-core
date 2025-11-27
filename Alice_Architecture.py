import numpy as np
import copy


# --- 1. CORE CONFIGURATION AND HYPERPARAMETERS ---

class Config:
    """Alice Architectureの初期値、次元、ハイパーパラメータを定義するクラス。"""

    # 次元定義
    N_C = 512       # 意味構造 (C)
    N_ESELF = 128   # 自己モデル (E_self)
    N_E_P = 64      # 制御負荷 (E_ctrl), 予測誤差 (P)
    N_M = 256       # 記憶 (M)
    N_ENV = 644     # 環境入力 (E_env = 512+128+4)
    N_HPZ = N_E_P   # 累積幸福 (H_pz)
    N_UPZ = N_E_P   # 累積不安 (U_pz)
    N_R = N_E_P     # 報酬/リスク (R)
    N_H = N_E_P     # 行動履歴 (H)
    N_VFL = N_E_P   # 価値関数層 (VFL)
    N_ES = N_E_P    # メタ認知 (E_s)
    N_EOBJ = N_E_P  # 客観化 (E_obj)
    N_RRL = N_M     # 関係再構築 (RRL)

    # A-TDL 学習安定化ハイパーパラメータ
    ETA_X = 1e-4        # 学習率 (W^X のベース更新率)
    CLIP_NORM = 5.0     # 最大勾配ノルム
    T_BPTT = 16         # BPTT 時間窓 (16ステップ遡及)
    GAMMA = 0.99        # 割引率 (G_Value)

    # 損失重み
    LAMBDA_P = 1.0      # 予測誤差 (P)
    LAMBDA_C = 0.5      # 制御負荷 (E_ctrl)
    LAMBDA_S = 0.8      # 自己整合 (E_self)

    # 進化則ハイパーパラメータ
    ETA_THETA = 1e-6    # 人格進化率
    THETA_MAX = 1.0     # 人格パラメータの上限/下限
    RHO_SNEL = 0.1      # SNEL信号駆動率 (ΔSNEL'にも適用)
    RHO_ISL = 0.1       # ISL信号駆動率
    ALPHA_PRED = 0.01   # E_self^pred の予測率
    KC_ISL = 1.0        # ISLにおけるE_ctrlの分散に対する抑制係数

    # ノイズ設定
    LAMBDA_ANNEAL = 0.001
    SIGMA2_MIN = 1e-4
    SIGMA2_INITIAL = {
        'C': 0.10, 'M': 0.10, 'RRL': 0.10, 'H': 0.10, 'VFL': 0.10, 'A': 0.20,
        'Es': 0.00, 'Eobj': 0.00, 'Ectrl': 0.00, 'Eself': 0.00, 'Hpz': 0.00, 'Upz': 0.00
    }

    # CSCハイパーパラメータ
    CSC_MAX_ITER = 5    # CSCの最大反復回数
    CSC_TOLERANCE = 1e-3 # 行動の収束許容誤差

# --- 1.5. 情動コアクラス (±0 Theory) の厳密化版 ---

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
        # 厳密化 1: 駆動源 R_t をベクトルのまま使用 (要素ごとのmax(0, x))
        drift_H = (1 - beta_H) * self.H_prime + alpha_H * np.maximum(R_t, 0) - gamma_HU * self.U_prime

        # 厳密化 2: ノイズ項の追加 (SDE)
        noise_H = self.SIGMA_H * np.random.randn(N_DIM)
        H_prime_next = np.maximum(0, drift_H + noise_H) # Hは非負

        # U' (不安) のダイナミクス
        drift_U = (1 - beta_U) * self.U_prime + alpha_U * np.maximum(P_t, 0) - gamma_UH * self.H_prime

        # 厳密化 2: ノイズ項の追加 (SDE)
        noise_U = self.SIGMA_U * np.random.randn(N_DIM)
        U_prime_next = np.maximum(0, drift_U + noise_U) # Uは非負

        # 状態の更新
        self.H_prime = H_prime_next
        self.U_prime = U_prime_next

        return H_prime_next, U_prime_next


# --- 2. 勾配計算とBPTTを担う補助クラス ---

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

            # E_ctrl -> Net_C への新しい伝播パス (恒常性逸脱フィードバック) 【厳密化】
            # dE_ctrl/dNet_C_change = f'(E_ctrl) * U_Ectrl_CNet
            # Net_C の変動を計算するために Net_C(t-k-1) が必要
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


# --- 3. ALICE ARCHITECTURE CORE CLASS ---

class AliceArchitecture:
    def __init__(self, config=Config()):
        self.config = config
        self.t = 0 # タイムステップ

        self._init_states()
        self._init_weights()
        self._init_theta()
        self._init_noise()

        # ±0理論コアの初期化 (統合)
        self.zero_one_core = ZeroOneTheory(config)

        # ATDL Learnerの初期化に必要な全重み辞書
        all_weights = {
            'W_C': self.W_C, 'U_C_Eenv': self.U_C_Eenv, 'U_C_M': self.U_C_M,
            'U_M_C': self.U_M_C, 'U_RRL_M': self.U_RRL_M, 'U_C_RRL': self.U_C_RRL,
            'U_H_A': self.U_H_A, 'U_VFL_R': self.U_VFL_R, 'U_VFL_S': self.U_VFL_S,
            'U_VFL_Hpz': self.U_VFL_Hpz, 'U_VFL_Upz': self.U_VFL_Upz, # 【追加】
            'U_Es_S': self.U_Es_S, 'U_Eobj_C': self.U_Eobj_C,
            'U_Eobj_M': self.U_Eobj_M, 'U_Ectrl_Es': self.U_Ectrl_Es,
            'U_Ectrl_CNet': self.U_Ectrl_CNet, # 【追加】
            'U_Eself_Eobj': self.U_Eself_Eobj,
            'U_NLG_C': self.U_NLG_C, 'U_NLG_M': self.U_NLG_M,
            'W_RRL': self.W_RRL, 'W_VFL': self.W_VFL
        }

        self.learner = ATDLLearner(self.config, all_weights, self.trainable_weights)


    # --- 補助関数 (init, noise, activation, etc.) ---
    def _init_states(self):
        # ... (状態初期化は変更なし) ...
        cfg = self.config
        self.C = np.zeros(cfg.N_C)
        self.E_ctrl = np.zeros(cfg.N_E_P)
        self.P = np.zeros(cfg.N_E_P)
        self.M = np.zeros(cfg.N_M)
        self.H_pz = np.zeros(cfg.N_HPZ)
        self.U_pz = np.zeros(cfg.N_UPZ)
        self.R = np.zeros(cfg.N_R)
        self.RRL = np.zeros(cfg.N_RRL)
        self.H = np.zeros(cfg.N_H)
        self.VFL = np.zeros(cfg.N_VFL)
        self.E_s = np.zeros(cfg.N_ES)
        self.E_obj = np.zeros(cfg.N_EOBJ)
        self.E_self_pred = np.zeros(cfg.N_ESELF)
        self.E_self = np.random.uniform(-1, 1, cfg.N_ESELF)
        self.Net_C = np.zeros(cfg.N_C)
        self.Net_A = np.zeros(1)
        self.U_prime_pz = np.zeros(cfg.N_UPZ)
        self.external_reward = 0.0

        self_state_names = ['C', 'E_ctrl', 'P', 'M', 'H_pz', 'U_pz', 'R', 'RRL', 'H', 'VFL', 'E_s', 'E_obj', 'E_self_pred', 'E_self', 'Net_C', 'Net_A', 'U_prime_pz']

        for name in self_state_names:
            setattr(self, f"{name}_next", getattr(self, name).copy())
        self.b_C = np.zeros(cfg.N_C)


    def _init_weights(self):
        def xavier_init(in_dim, out_dim):
            scale = 1.0 / np.sqrt(in_dim)
            return np.random.randn(out_dim, in_dim) * scale
        cfg = self.config

        # 学習対象 W^X (知性) の重み
        self.W_C = xavier_init(cfg.N_C, cfg.N_C) * np.sqrt(cfg.N_C) # 自己回帰
        self.U_C_Eenv = xavier_init(cfg.N_ENV, cfg.N_C) # 環境入力
        self.U_NLG_C = xavier_init(cfg.N_C, 1) # 行動/出力へ

        self.trainable_weights = {
            'W_C': self.W_C, 'U_C_Eenv': self.U_C_Eenv, 'U_NLG_C': self.U_NLG_C
        }

        # その他の重み
        self.U_C_M = xavier_init(cfg.N_M, cfg.N_C)
        self.U_M_C = xavier_init(cfg.N_C, cfg.N_M)
        self.U_RRL_M = xavier_init(cfg.N_M, cfg.N_RRL)
        self.U_C_RRL = xavier_init(cfg.N_RRL, cfg.N_C)
        self.U_H_A = xavier_init(1, cfg.N_H)
        self.U_VFL_R = xavier_init(cfg.N_R, cfg.N_VFL)
        self.U_VFL_S = xavier_init(cfg.N_ENV + cfg.N_C, cfg.N_VFL)
        self.U_VFL_Hpz = xavier_init(cfg.N_HPZ, cfg.N_VFL)
        self.U_VFL_Upz = xavier_init(cfg.N_UPZ, cfg.N_VFL) # 【追加】
        self.U_Es_S = xavier_init(cfg.N_ENV + cfg.N_C + cfg.N_M, cfg.N_ES)
        self.U_Eobj_C = xavier_init(cfg.N_C, cfg.N_EOBJ)
        self.U_Eobj_M = xavier_init(cfg.N_M, cfg.N_EOBJ)
        self.U_Ectrl_Es = xavier_init(cfg.N_ES, cfg.N_E_P)
        self.U_Ectrl_CNet = xavier_init(cfg.N_C, cfg.N_E_P) # 【追加】
        self.U_Eself_Eobj = xavier_init(cfg.N_EOBJ, cfg.N_ESELF)
        self.U_NLG_M = xavier_init(cfg.N_M, 1)
        self.W_RRL = xavier_init(cfg.N_RRL, cfg.N_RRL)
        self.W_VFL = xavier_init(cfg.N_VFL, cfg.N_VFL)

    def _init_theta(self):
        # 人格パラメータ θ
        self.theta = {'H_base': 0.0, 'U_base': 0.0, 'beta_H': 0.1, 'beta_U': 0.1,
                      'gamma_HU': 0.5, 'gamma_UH': 0.5, 'kappa_U': 1.0,
                      'alpha_H': 0.1, 'alpha_U': 0.1, 'theta_R': 0.5}


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
        # 簡易LLM埋め込み代替ロジック (変更なし)
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

        # 過去のC層活性化前入力 Net_C を保存 (E_ctrl の厳密化に必要)
        Net_C_prev = self.Net_C.copy()

        # --- (1) 認知層の更新 (C, M, RRL, P) ---

        # C(t+1)
        Net_C_next = (
            self.W_C @ self.C +
            self.U_C_Eenv @ E_env_t +
            self.U_C_M @ self.M +
            self.b_C
        )
        self.C_next = f_X(Net_C_next) + self._add_noise('C', cfg.N_C)
        self.Net_C_next = Net_C_next

        # M(t+1), R(t+1), RRL(t+1) の計算は変更なし (省略)
        alpha_M = 0.5
        self.M_next = (1 - alpha_M) * self.M + alpha_M * f_X(self.U_M_C @ self.C) + self._add_noise('M', cfg.N_M)
        alpha_R = 0.1
        self.R_next = (1 - alpha_R) * self.R + alpha_R * self.external_reward * np.ones_like(self.R) + self._add_noise('R', cfg.N_R)
        self.RRL_next = f_X(self.W_RRL @ self.RRL + self.U_RRL_M @ self.M) + self._add_noise('RRL', cfg.N_RRL)

        # P(t+1) (予測誤差)
        self.P_next = self.C_next - self.U_C_RRL @ self.RRL_next

        # H(t+1) (行動履歴)
        alpha_H_Hist = 0.1
        self.H_next = (1 - alpha_H_Hist) * self.H + alpha_H_Hist * f_X(self.U_H_A @ np.array([A_t])) + self._add_noise('H', cfg.N_H)

        # --- (2) 情動コアの接続 (H_pz, U_pz) ---

        # ±0理論コアを実行し、結果を H_pz_next / U_pz_next に直接代入する (厳密化 ZeroOneTheory)
        H_prime_result, U_prime_result = self.zero_one_core.step(self.R, self.P, self.theta)

        self.H_pz_next = H_prime_result
        self.U_pz_next = U_prime_result

        # 補正後不安 U'_pz (V(t)計算用)
        R_val = self._recovery_R(self.U_pz_next, self.H_pz_next)
        self.U_prime_pz_next = self.U_pz_next - R_val

        # --- (3) 自己モデル層の更新 (E_s, E_obj, E_ctrl, E_self) ---

        # (9.1) Metacognition Layer E_s(t+1)
        S_t = np.concatenate([E_env_t, self.C, self.M])
        self.E_s_next = f_X(self.U_Es_S @ S_t) + self._add_noise('Es', cfg.N_ES)

        # (9.2) Objectification Layer E_obj(t+1)
        self.E_obj_next = f_X(self.U_Eobj_C @ self.C + self.U_Eobj_M @ self.M) + self._add_noise('Eobj', cfg.N_EOBJ)

        # (9.3) Integrated Control Layer E_ctrl(t+1) 【厳密化】
        Net_C_change = self.Net_C_next - Net_C_prev

        self.E_ctrl_next = f_X(
            self.U_Ectrl_Es @ self.E_s +                 # E_sからのメタレベル制御入力
            self.U_Ectrl_CNet @ Net_C_change             # C層の入力変動による不安定性反映
        ) + self._add_noise('Ectrl', cfg.N_E_P)

        # (9.4) Self-Model Layer E_self(t+1)
        self.E_self_next = f_X(self.U_Eself_Eobj @ self.E_obj) + self._add_noise('Eself', cfg.N_ESELF)

        # (11) Self-Prediction Model E_self^pred(t+1)
        alpha_pred = cfg.ALPHA_PRED
        self.E_self_pred_next = (1 - alpha_pred) * self.E_self_pred + alpha_pred * self.E_self + self._add_noise('Eself_pred', cfg.N_ESELF)

        # --- (4) 価値関数層の更新 (VFL) ---

        S_prime_t = np.concatenate([E_env_t, self.C])

        # (8) Value Function Layer VFL(t+1) 【厳密化】
        Net_VFL_next = (
            self.W_VFL @ self.VFL +
            self.U_VFL_R @ self.R +
            self.U_VFL_S @ S_prime_t +
            self.U_VFL_Hpz @ self.H_pz -             # 幸福コアからの正の入力
            self.U_VFL_Upz @ self.U_pz               # 【追加】不安コアからの負の入力
        )
        self.VFL_next = f_X(Net_VFL_next) + self._add_noise('VFL', cfg.N_VFL)

    def _save_history(self, E_env_t, A_t):
        # ... (履歴保存は変更なし) ...
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
        # ... (状態コミットは変更なし) ...
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

    def _calculate_V_from_state(self, state_prefix):
        # ... (V(t)計算は変更なし) ...
        cfg = self.config
        theta = self.theta

        VFL = getattr(self, state_prefix + 'VFL')
        U_prime_pz = getattr(self, state_prefix + 'U_prime_pz')
        P = getattr(self, state_prefix + 'P')
        E_ctrl = getattr(self, state_prefix + 'E_ctrl')
        E_self = getattr(self, state_prefix + 'E_self')
        E_self_pred = getattr(self, state_prefix + 'E_self_pred')
        H = getattr(self, state_prefix + 'H')
        R = getattr(self, state_prefix + 'R')

        V_value = np.sum(VFL)
        max_U_prime = np.max(U_prime_pz)

        # L_P (予測誤差コスト)
        L_P = cfg.LAMBDA_P * np.sum(P) * (1.0 + theta['kappa_U'] * max_U_prime)
        # L_C (制御負荷コスト)
        L_C = cfg.LAMBDA_C * np.var(E_ctrl)
        # L_S (自己整合コスト)
        L_S = cfg.LAMBDA_S * np.sum((E_self - E_self_pred)**2)

        V_t = V_value - L_P - L_C - L_S

        V_terms = {
            'V_t': V_t, 'Dist_self': L_S, 'Var_ctrl': np.var(E_ctrl),
            'sum_VFL': V_value, 'Will_signal': self._f_Will(H, R)
        }
        return V_t, V_terms


    def _calculate_V_next_state(self):
        return self._calculate_V_from_state('_next')


    def _calculate_V_current_state(self):
        return self._calculate_V_from_state('')


    def _run_counterfactual_simulation(self, E_env_t, A_simul):
        # ... (反実仮想は変更なし) ...
        virtual_alice = copy.deepcopy(self)
        virtual_alice.external_reward = self.external_reward
        virtual_alice._update_layers(E_env_t, A_simul)
        V_simul, _ = virtual_alice._calculate_V_next_state()
        return V_simul


    def _evolve_theta(self, V_t, V_terms, E_env_t):
        """
        人格パラメータ θ の進化則 (SNEL/ISL + ΔSNEL'の統合) - 【厳密化】
        """
        cfg = self.config
        theta = self.theta

        # 厳密化: U_pzの平均値を取得 (不安駆動の加速に使用)
        mean_U_pz = np.mean(self.U_pz)

        # --- (1) SNEL / ISL の計算 ---

        # SNEL (Self-Non-Equilibrium Loss) 【厳密化】
        Dist_self = V_terms['Dist_self']
        Will_signal = V_terms['Will_signal']
        # 不安が強いほど進化を加速する項 (1 + mean_U_pz^2) を導入
        Delta_SNEL = cfg.RHO_SNEL * Dist_self * Will_signal * (1.0 + mean_U_pz**2)

        # ISL (Inertial Stability Loss) 【厳密化】
        Var_ctrl = V_terms['Var_ctrl']
        sum_VFL = V_terms['sum_VFL']

        # 厳密な満足度比率 (VFLに対するV_tの比率: コストが低いほど高い)
        satisfaction_ratio = V_t / (sum_VFL + 1e-6)
        Delta_ISL = cfg.RHO_ISL * np.exp(-cfg.KC_ISL * Var_ctrl) * satisfaction_ratio

        # --- (2) ΔSNEL' (反実仮想信号) の計算 ---

        A_simul = -V_terms['Will_signal']
        V_simul = self._run_counterfactual_simulation(E_env_t, A_simul)

        Delta_SNEL_prime = V_simul - V_t

        # --- (3) 進化信号の統合と更新 ---

        Total_SNEL = Delta_SNEL + cfg.RHO_SNEL * Delta_SNEL_prime

        updates = {
            'beta_U': Total_SNEL, 'alpha_U': Total_SNEL, 'gamma_HU': -Total_SNEL,
            'kappa_U': Total_SNEL,
            'beta_H': -Delta_ISL, 'H_base': Delta_ISL,
            'alpha_H': Delta_ISL, 'gamma_UH': Delta_ISL
        }

        for key, delta in updates.items():
            theta_t = theta.get(key, 0.0)
            divergence_prevention = (1.0 - abs(theta_t) / cfg.THETA_MAX)
            theta_next = theta_t + cfg.ETA_THETA * delta * divergence_prevention
            theta[key] = np.clip(theta_next, -cfg.THETA_MAX, cfg.THETA_MAX)

        return Delta_SNEL_prime


    def _run_csc_stabilization(self, E_env_t, A_t_initial):
        # ... (CSCは変更なし) ...
        A_prev = A_t_initial

        for i in range(self.config.CSC_MAX_ITER):
            Net_A_next = self.U_NLG_C @ self.C_next + self.U_NLG_M @ self.M_next
            A_current = np.tanh(Net_A_next)[0]

            if np.abs(A_current - A_prev) < self.config.CSC_TOLERANCE and i > 0:
                break

            A_prev = A_current

        self.Net_A_next = Net_A_next
        A_final_refined = A_prev + self._add_noise('A', 1)[0]
        return np.clip(A_final_refined, -1.0, 1.0)

    def step(self, user_input_text: str, external_reward: float):
        # ... (step関数は変更なし) ...
        self.t += 1
        self.external_reward = external_reward
        cfg = self.config

        E_token = self._generate_E_token(user_input_text)
        E_context = np.random.randn(cfg.N_ESELF) * 0.1
        E_scalar = np.array([external_reward, self.t % 100, self.t, 1 if np.random.rand() < 0.1 else 0])
        E_env_t = np.concatenate([E_token, E_context, E_scalar])

        Net_A_t = self.U_NLG_C @ self.C + self.U_NLG_M @ self.M
        A_t_initial = np.tanh(Net_A_t)[0] + self._add_noise('A', 1)[0]
        A_t_initial = np.clip(A_t_initial, -1.0, 1.0)

        self._save_history(E_env_t, A_t_initial)
        self._update_layers(E_env_t, A_t_initial)

        V_provisional_t_plus_1, _ = self._calculate_V_next_state()

        A_final_refined = self._run_csc_stabilization(E_env_t, A_t_initial)

        self._commit_state()

        V_t, V_terms = self._calculate_V_current_state()

        Delta_SNEL_prime = self._evolve_theta(V_t, V_terms, E_env_t)

        TD_error = self.learner.learn_step(V_provisional_t_plus_1, V_t, self.learner.history[-1], self.theta)

        Var_ctrl = V_terms['Var_ctrl']
        tau_ctrl = 0.05
        is_stable = Var_ctrl <= tau_ctrl

        output = {
            'action': A_final_refined,
            'V_total': V_t,
            'happiness_core': np.mean(self.H_pz),
            'uncertainty_core': np.mean(self.U_pz),
            'control_load': Var_ctrl,
            'is_stable': is_stable,
            'theta_snapshot': self.theta,
            'TD_error': TD_error,
            'Delta_SNEL_prime': Delta_SNEL_prime
        }

        return output
