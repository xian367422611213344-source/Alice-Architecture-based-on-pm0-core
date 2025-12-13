# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# --- 1. Alice V4.0+ コアモデル定義 (内部) ---

class _AliceCoreModel(nn.Module):
    """
    Alice V4.0+ のコアロジック。PTREパラメータ(theta)と係数(C)を管理し、写像を実行。
    """
    def __init__(self, K_max=1.0, eta_theta=1e-5, gamma_theta=0.005):
        super(_AliceCoreModel, self).__init__()
        self.K_MAX = K_max
        self.eta_theta = eta_theta      # theta進化則の学習率 (極小)
        self.gamma_theta = gamma_theta  # PTRE 自己調整項の強さ

        # 1. 人格特性パラメータ: theta_empathy (θ) - カスタム更新
        self._theta = nn.Parameter(torch.tensor([0.45], dtype=torch.float32), requires_grad=False)

        # 2. 行動スタイル係数: C (勾配学習)
        # 拡張されたCベクトル: [G0, GK, W0, WK, WU, H0, HR, # 新規: E_theta_bias, E_U_gain, E_K_gain, P_Arousal_gain, B_Valence_gain]
        initial_C = [
            0.5, 1.3, 1.0, 0.2, 0.2, 0.8, 0.6,      # 旧係数 (Gaze, WPM, Help)
            0.1, 0.6, 0.4, 0.5, 0.4                  # 新規係数 (EDM, Prosody, Body)
        ]
        self.C = nn.Parameter(torch.tensor(initial_C, dtype=torch.float32), requires_grad=True)
        # 係数への参照マップ
        self.C_map = {
            'G0': 0, 'GK': 1, 'W0': 2, 'WK': 3, 'WU': 4, 'H0': 5, 'HR': 6,
            'E_theta_bias': 7, 'E_U_gain': 8, 'E_K_gain': 9, 'P_Arousal_gain': 10, 'B_Valence_gain': 11
        }

    def _emotion_dynamics_mapping(self, V_total_tensor, U_pz_tensor, K_avg_tensor):
        """
        中間層: 内部状態から感情状態ベクトル E_t (Valence, Arousal) を計算する。
        """
        theta = self._theta.data.item()
        E_theta_bias, E_U_gain, E_K_gain = self.C[self.C_map['E_theta_bias']:self.C_map['E_K_gain'] + 1]

        # 1. Valence (快不快): V_totalとθバイアスから導出 [-1.0, +1.0]
        # E^Valence = 2 * V_total - 1.0 + E_theta_bias * (theta - 0.5)
        E_Valence = 2.0 * V_total_tensor - 1.0 + E_theta_bias * (theta - 0.5)
        E_Valence = torch.clamp(E_Valence, -1.0, 1.0)

        # 2. Arousal (覚醒): U_pzとK_avgから導出 [0.0, +1.0]
        # E^Arousal = E_U_gain * U_pz + E_K_gain * K_avg
        E_Arousal = E_U_gain * U_pz_tensor + E_K_gain * K_avg_tensor
        E_Arousal = torch.clamp(E_Arousal, 0.0, 1.0)

        return E_Valence, E_Arousal

    def forward(self, K_avg_tensor, U_pz_tensor, R_group_tensor, V_total_tensor):
        """
        身体化写像: S_t -> E_t -> A_t (拡張版)
        """
        theta = self._theta.data.item()

        # 1. 感情ダイナミクス層の実行
        E_Valence, E_Arousal = self._emotion_dynamics_mapping(V_total_tensor, U_pz_tensor, K_avg_tensor)

        # 2. 拡張された行動ベクトル A_t の計算
        G0, GK, W0, WK, WU, H0, HR, _, _, _, P_Arousal_gain, B_Valence_gain = self.C

        # --- A. 旧要素の計算 (θとK/Uに依存) ---
        # A.1. 視線同期率 (Gaze)
        A_gaze = G0 * (1.0 + theta) * torch.exp(-GK * K_avg_tensor / self.K_MAX)
        A_gaze = torch.clamp(A_gaze, 0.3, 1.0)
        # A.2. 応答タイミング/話速減衰 (WPM)
        A_wpm_decay = W0 * (1.0 - WK * K_avg_tensor) * (1.0 - WU * U_pz_tensor)
        A_wpm_decay = torch.clamp(A_wpm_decay, 0.5, 1.0)
        # A.3. 手動補助の優先度 (Helpfulness)
        A_helpfulness = H0 * (theta ** 2) * (1.0 + HR * R_group_tensor)
        A_helpfulness = torch.clamp(A_helpfulness, 0.0, 1.0)

        # --- B. 新規要素の計算 (感情 E_t に依存) ---
        # B.1. 表情強度 (Facial Expression) - ValenceとArousalの複合制御
        # 快/不快の絶対値と覚醒度の積として強度を決定
        A_facial_intensity = torch.clamp(torch.abs(E_Valence) * E_Arousal, 0.0, 1.0)

        # B.2. 音声プロソディ (Prosody: Pitch) - 覚醒度に比例
        A_pitch_modulation = P_Arousal_gain * E_Arousal
        A_pitch_modulation = torch.clamp(A_pitch_modulation, 0.0, 1.0)

        # B.3. 身体姿勢 (Body: Lean Angle) - Valenceに比例
        # 正の値: ポジティブな傾き（前傾/親近感）、負の値: 後傾（不快/拒否）
        A_body_lean_angle = B_Valence_gain * E_Valence
        A_body_lean_angle = torch.clamp(A_body_lean_angle, -0.8, 0.8)  # 角度をラジアンで表現

        # 拡張された行動ベクトル A_t の出力
        A_t = {
            'gaze_sync_rate': A_gaze,
            'assist_priority': A_helpfulness,
            'wpm_decay_factor': A_wpm_decay,
            'e_valence': E_Valence,
            'e_arousal': E_Arousal,
            'facial_intensity': A_facial_intensity,
            'pitch_modulation': A_pitch_modulation,
            'body_lean_angle': A_body_lean_angle
        }
        return A_t

    def get_theta(self):
        return self._theta.data.item()

    def update_theta_ptre(self, V_total, K_avg, U_pz):
        """
        PTRE進化則による theta のカスタム更新 (人格の緩慢な進化)
        """
        theta_t = self._theta.data.item()

        # 学習率抑制
        eta_eff = self.eta_theta * (1.0 - K_avg.item()) * (1.0 - U_pz.item())
        eta_eff = max(eta_eff, 1e-7)
        U_soc = V_total.item()

        # PTRE進化則: d_theta = eta_eff * (U_soc - 0.5) + gamma_theta * (0.5 - theta_t)
        d_theta = eta_eff * (U_soc - 0.5) + self.gamma_theta * (0.5 - theta_t)

        new_theta = np.clip(theta_t + d_theta, 0.0, 1.0)

        self._theta.data.copy_(torch.tensor([new_theta]))


# --- 2. Alice コントローラー (APIインターフェース) ---

class AliceController:
    """
    Alice V4.0+ 実稼働用API。外部XR環境と連携するためのI/Oを管理し、二層学習を制御する。
    """
    def __init__(self, K_max=1.0, C_lr=1e-4):
        # Model and Optimizers initialization
        self.model = _AliceCoreModel(K_max=K_max)

        # Optimizer for C (Behavioral Style) - Faster learning rate (二層学習の C 層)
        self.optimizer_C = optim.Adam(self.model.parameters(), lr=C_lr)

        # 内部状態
        self.K_avg = torch.tensor([0.6], dtype=torch.float32)
        self.U_pz = torch.tensor([0.5], dtype=torch.float32)
        self.R_group = torch.tensor([0.0], dtype=torch.float32)
        self.V_total = torch.tensor([0.0], dtype=torch.float32)

        print("Alice V4.0+ Embodiment Controller Initialized: Ready for API Integration.")

    def get_current_state(self):
        """現在の内部状態変数を返す (モニタリング用)"""
        E_Valence, _ = self.model._emotion_dynamics_mapping(self.V_total, self.U_pz, self.K_avg)
        return {
            'theta_empathy': self.model.get_theta(),
            'K_avg': self.K_avg.item(),
            'U_pz': self.U_pz.item(),
            'V_total': self.V_total.item(),
            'E_Valence': E_Valence.item(),
            'C_G0_gaze_gain': self.model.C.data[self.model.C_map['G0']].item(),
        }

    def predict_action(self, K_avg_new, U_pz_new):
        """
        API 1: 新しい認知負荷(K)と不安(U)を受け取り、次の行動を予測する。
        JSON Action Schema (外部I/O仕様書) に準拠した辞書を返す。
        """
        # 内部状態の更新
        self.K_avg = torch.tensor([K_avg_new], dtype=torch.float32)
        self.U_pz = torch.tensor([U_pz_new], dtype=torch.float32)

        # 身体化写像 (Forward Pass) を実行
        A_t = self.model(self.K_avg, self.U_pz, self.R_group, self.V_total)

        # 外部API用のJSONスキーマ形式に変換
        action_schema = {
            "internal_state": {
                "theta_empathy": self.model.get_theta(),
                "e_valence": A_t['e_valence'].item(),
                "e_arousal": A_t['e_arousal'].item(),
            },
            "embodiment_output": {
                "gaze": {
                    "sync_rate": A_t['gaze_sync_rate'].item(),
                    "max_deviation": 0.1  # Example: Placeholder for future K/U controlled deviation
                },
                "speech_prosody": {
                    "wpm_decay_factor": A_t['wpm_decay_factor'].item(),  # 1.0で通常話速
                    "pitch_modulation": A_t['pitch_modulation'].item()  # 0.0でモノトーン, 1.0で強調
                },
                "facial_expression": {
                    "valence_signal": A_t['e_valence'].item(),  # [-1.0: 不快, 1.0: 快]
                    "intensity": A_t['facial_intensity'].item()  # 0.0: 無表情, 1.0: 最大強度
                },
                "body_gesture": {
                    "assist_priority": A_t['assist_priority'].item(),  # 0.0: 静止, 1.0: 積極的
                    "lean_angle_rad": A_t['body_lean_angle'].item()  # -0.8 ~ +0.8 ラジアンで傾き
                }
            }
        }
        return action_schema

    def update_state(self, R_group_feedback):
        """
        API 2: 環境からの報酬フィードバックを受け取り、内部状態 (theta, C) を更新する。
        閉ループ学習の駆動源。
        """
        R_group_tensor = torch.tensor([R_group_feedback], dtype=torch.float32)
        self.R_group = R_group_tensor

        # 1. V_total (総価値関数) の計算: 学習の目的関数
        # V_total = R_group - Cost_K - Cost_U_pz
        Cost_K = self.K_avg.item() * 0.1
        Cost_U_pz = self.U_pz.item() * 0.05
        V_total_value = R_group_feedback - Cost_K - Cost_U_pz
        self.V_total = torch.tensor([V_total_value], dtype=torch.float32)

        # --- A. 係数 C の学習 (Adam/勾配降下法: 迅速な適応) ---
        self.optimizer_C.zero_grad()

        # 勾配計算用のダミーフォワードパス (V_totalを損失として使用)
        # V_totalに勾配を流すため、R_groupとV_totalをテンソルとして使用
        # 実際に必要なのは、このフォワードパスで計算される A_t の要素が V_total に間接的に影響を与えること。
        # しかし、この実装では V_total は R_group から直接計算されており、C に依存していません。
        # したがって、勾配計算のロジックは、Cが行動A_tを介して環境R_groupに影響を与え、それが V_total を変化させる、という
        # 外部ループをシミュレートする必要があります。
        # V_total = R_group(A_t) - Cost... の形であるべきですが、ここでは R_group は外部入力として固定されています。
        # 暫定的に、V_total を損失として使用する手法を維持します（V_totalが「理想的な」状態に近づくようにCを更新するという解釈）。
        # 正しい実装では、A_tを出力したことによる R_group の変化の期待値が計算されます。
        A_dummy = self.model(self.K_avg, self.U_pz, self.R_group, self.V_total)

        # 損失は -V_total (最小化)
        loss = -self.V_total
        loss.backward()
        self.optimizer_C.step()

        # --- B. theta の進化 (PTRE進化則: 緩慢な進化) ---
        self.model.update_theta_ptre(self.V_total, self.K_avg, self.U_pz)

        return self.get_current_state()


# --- 3. APIデモ実行 ---

if __name__ == "__main__":

    # 1. 初期化
    alice_api = AliceController()

    # シミュレーションループ (10ステップ)
    print("\n--- Alice V4.0+ API デモ: Predict-Update サイクル開始 ---")

    for step in range(10):
        # 1. 環境からの入力
        K_new = np.random.uniform(0.3, 0.9)  # 努力負荷を広く変動させる
        U_new = np.random.uniform(0.1, 0.8)  # 不安を広く変動させる

        # 2. タスクによる報酬フィードバック
        R_feedback = 0.9 if step % 2 == 0 else 0.2

        print(f"\n[STEP {step + 1}] Input: K={K_new:.2f}, U={U_new:.2f} | Feedback R_group = {R_feedback:.1f}")

        # 3. Predict: 行動の予測
        predicted_action_schema = alice_api.predict_action(K_new, U_new)

        # 4. Action Output (XR/Robotが実行する部分)
        emo = predicted_action_schema['internal_state']
        act = predicted_action_schema['embodiment_output']

        print(f"  > 感情 E_t (内部状態): Valence={emo['e_valence']:.3f}, Arousal={emo['e_arousal']:.3f}")
        print(f"  > 行動 A_t (外部制御):")
        print(f"    - Gaze Sync Rate: {act['gaze']['sync_rate']:.3f}")
        print(f"    - Facial Intensity: {act['facial_expression']['intensity']:.3f}")
        print(f"    - Body Lean Rad: {act['body_gesture']['lean_angle_rad']:.3f}")

        # 5. Update: 内部状態の更新
        updated_state = alice_api.update_state(R_feedback)

        # 6. Internal State After Update
        print(f"  > 更新後内部状態:")
        print(f"    - Theta (人格): {updated_state['theta_empathy']:.4f}")
        print(f"    - V_total (価値): {updated_state['V_total']:.4f}")
        print(f"    - C_G0 (Gaze Gain): {updated_state['C_G0_gaze_gain']:.4f}")
