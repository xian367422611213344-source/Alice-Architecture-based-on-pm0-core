# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# --- 1. Alice V4.0+ 身体化写像モデル定義 (Embodiment Layer) ---

class _AliceCoreModel(nn.Module):
    """
    Alice V4.0+ の身体化写像層 (Embodiment Layer)。
    行動スタイル係数(C)のみを管理し、
    Coreからの入力(K, U, V, R, theta)を外部行動に写像する。
    """
    def __init__(self, K_max=1.0):
        super(_AliceCoreModel, self).__init__()
        self.K_MAX = K_max

        # 2. 行動スタイル係数: C (勾配学習) - 維持
        # 拡張されたCベクトル: [G0, GK, W0, WK, WU, H0, HR, E_theta_bias, E_U_gain, E_K_gain, P_Arousal_gain, B_Valence_gain]
        initial_C = [
            0.5, 1.3, 1.0, 0.2, 0.2, 0.8, 0.6,      # 旧係数 (Gaze, WPM, Help)
            0.1, 0.6, 0.4, 0.5, 0.4                  # 新規係数 (EDM, Prosody, Body)
        ]
        self.C = nn.Parameter(torch.tensor(initial_C, dtype=torch.float32), requires_grad=True)
        
        # 係数への参照マップ
        self.C_map = {
            'G0': 0, 'GK': 1, 'W0': 2, 'WK': 3, 'WU': 4, 'H0': 5, 'HR': 6,
            'E_theta_bias': 7, 'E_U_gain': 8, 'E_K_gain': 9, 
            'P_Arousal_gain': 10, 'B_Valence_gain': 11
        }

    def _emotion_dynamics_mapping(self, V_total_tensor, U_pz_tensor, K_avg_tensor, theta_current):
        """
        中間層: 内部状態から感情状態ベクトル E_t (Valence, Arousal) を計算する。
        """
        theta = theta_current # Coreからの入力を利用
        E_theta_bias, E_U_gain, E_K_gain = self.C[self.C_map['E_theta_bias']:self.C_map['E_K_gain'] + 1]

        # 1. Valence (快不快): V_totalとθバイアスから導出 [-1.0, +1.0]
        E_Valence = 2.0 * V_total_tensor - 1.0 + E_theta_bias * (theta - 0.5)
        E_Valence = torch.clamp(E_Valence, -1.0, 1.0)

        # 2. Arousal (覚醒): U_pzとK_avgから導出 [0.0, +1.0]
        E_Arousal = E_U_gain * U_pz_tensor + E_K_gain * K_avg_tensor
        E_Arousal = torch.clamp(E_Arousal, 0.0, 1.0)
        
        return E_Valence, E_Arousal

    def forward(self, K_avg_tensor, U_pz_tensor, R_group_tensor, V_total_tensor, theta_current):
        """
        身体化写像: S_t -> E_t -> A_t (拡張版)
        """
        theta = theta_current # Coreからの入力を利用
        
        # 1. 感情ダイナミクス層の実行
        E_Valence, E_Arousal = self._emotion_dynamics_mapping(V_total_tensor, U_pz_tensor, K_avg_tensor, theta)
        
        # 2. 拡張された行動ベクトル A_t の計算
        G0, GK, W0, WK, WU, H0, HR, _, _, _, P_Arousal_gain, B_Valence_gain = self.C
        
        # --- A. 旧要素の計算 (θとK/Uに依存) ---
        # Gaze Sync Rate
        A_gaze = G0 * (1.0 + theta) * torch.exp(-GK * K_avg_tensor / self.K_MAX)
        A_gaze = torch.clamp(A_gaze, 0.3, 1.0)
        
        # WPM Decay Factor
        A_wpm_decay = W0 * (1.0 - WK * K_avg_tensor) * (1.0 - WU * U_pz_tensor)
        A_wpm_decay = torch.clamp(A_wpm_decay, 0.5, 1.0)
        
        # Assist Priority (Helpfulness)
        A_helpfulness = H0 * (theta ** 2) * (1.0 + HR * R_group_tensor)
        A_helpfulness = torch.clamp(A_helpfulness, 0.0, 1.0)

        # --- B. 新規要素の計算 (感情 E_t に依存) ---
        # Facial Intensity
        A_facial_intensity = torch.clamp(torch.abs(E_Valence) * E_Arousal, 0.0, 1.0)
        
        # Pitch Modulation
        A_pitch_modulation = P_Arousal_gain * E_Arousal
        A_pitch_modulation = torch.clamp(A_pitch_modulation, 0.0, 1.0)
        
        # Body Lean Angle
        A_body_lean_angle = B_Valence_gain * E_Valence
        A_body_lean_angle = torch.clamp(A_body_lean_angle, -0.8, 0.8)

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


# --- 2. Alice コントローラー (APIインターフェース) ---

class AliceController:
    """
    Alice V4.0+ 実稼働用API。外部Coreから状態を受け取り、
    身体化写像とC係数（行動スタイル）の適応学習のみを担う。
    """
    def __init__(self, K_max=1.0, C_lr=1e-4):
        self.model = _AliceCoreModel(K_max=K_max)
        
        # C 係数 (行動スタイル) のための最適化
        self.optimizer_C = optim.Adam(self.model.parameters(), lr=C_lr)
        
        # 内部状態 (Coreから受け取った最新の値をキャッシュ/モニタリング用)
        self.K_avg = 0.6
        self.U_pz = 0.5
        self.R_group = 0.0
        self.V_total = 0.0
        self.theta_empathy = 0.45 # Coreから受け取った最新のThetaをキャッシュ
        
        print("Alice V4.0+ Embodiment Controller Initialized: Ready for API Integration.")

    def get_current_state(self):
        """現在の内部状態変数を返す (モニタリング用)"""
        # 現在キャッシュされている値を用いて感情状態を再計算
        E_Valence_tensor, _ = self.model._emotion_dynamics_mapping(
            torch.tensor([self.V_total], dtype=torch.float32), 
            torch.tensor([self.U_pz], dtype=torch.float32), 
            torch.tensor([self.K_avg], dtype=torch.float32), 
            self.theta_empathy
        )
        return {
            'theta_empathy': self.theta_empathy,
            'K_avg': self.K_avg,
            'U_pz': self.U_pz,
            'V_total': self.V_total,
            'E_Valence': E_Valence_tensor.item(),
            'C_G0_gaze_gain': self.model.C.data[self.model.C_map['G0']].item(),
        }

    def predict_action(self, K_avg_new, U_pz_new, theta_current, V_total_current, R_group_current):
        """
        API 1: Coreからの最新状態 (K, U, theta, V_total, R_group) を受け取り、行動を予測する。
        """
        # 内部状態のキャッシュを更新
        self.K_avg = K_avg_new
        self.U_pz = U_pz_new
        self.V_total = V_total_current
        self.R_group = R_group_current
        self.theta_empathy = theta_current

        # Tensorに変換
        K_avg_tensor = torch.tensor([K_avg_new], dtype=torch.float32)
        U_pz_tensor = torch.tensor([U_pz_new], dtype=torch.float32)
        V_total_tensor = torch.tensor([V_total_current], dtype=torch.float32)
        R_group_tensor = torch.tensor([R_group_current], dtype=torch.float32)
        
        # 身体化写像 (Forward Pass) を実行
        A_t = self.model(K_avg_tensor, U_pz_tensor, R_group_tensor, V_total_tensor, theta_current)
        
        # 外部API用のJSONスキーマ形式に変換
        action_schema = {
            "internal_state": {
                "theta_empathy": theta_current,
                "e_valence": A_t['e_valence'].item(),
                "e_arousal": A_t['e_arousal'].item(),
            },
            "embodiment_output": {
                "gaze": {
                    "sync_rate": A_t['gaze_sync_rate'].item(),
                    "max_deviation": 0.1
                },
                "speech_prosody": {
                    "wpm_decay_factor": A_t['wpm_decay_factor'].item(),
                    "pitch_modulation": A_t['pitch_modulation'].item()
                },
                "facial_expression": {
                    "valence_signal": A_t['e_valence'].item(),
                    "intensity": A_t['facial_intensity'].item()
                },
                "body_gesture": {
                    "assist_priority": A_t['assist_priority'].item(),
                    "lean_angle_rad": A_t['body_lean_angle'].item()
                }
            }
        }
        return action_schema

    def update_state(self, R_group_feedback, V_total_canonical, K_avg_current, U_pz_current, theta_current):
        """
        API 2: 環境からの報酬と Core が計算した V_total/theta を受け取り、
        係数 C の適応学習のみを実行する。
        """
        # 内部状態のキャッシュを更新
        self.R_group = R_group_feedback
        self.V_total = V_total_canonical
        self.K_avg = K_avg_current
        self.U_pz = U_pz_current
        self.theta_empathy = theta_current # Full Coreで進化した最新のThetaをキャッシュ

        R_group_tensor = torch.tensor([R_group_feedback], dtype=torch.float32)
        V_total_tensor = torch.tensor([V_total_canonical], dtype=torch.float32)
        K_avg_tensor = torch.tensor([K_avg_current], dtype=torch.float32)
        
        # --- 係数 C の学習 (Adam/勾配降下法: 迅速な適応) ---
        self.optimizer_C.zero_grad()
        
        # V_totalを損失として使用するためのフォワードパス (学習時の状態値を使用)
        # ネットワーク全体を計算し、グラフを構築する
        A_dummy = self.model(K_avg_tensor, torch.tensor([U_pz_current], dtype=torch.float32), R_group_tensor, V_total_tensor, theta_current)
        
        # 損失は -V_total (V_totalを最大化するため、最小化する損失関数として負の値を使用)
        loss = -V_total_tensor
        loss.backward()
        self.optimizer_C.step()

        # --- theta の進化 ---
        # 削除: Full Coreが責任を負うため、ここでは学習も進化も行わない。
        
        return self.get_current_state()


# --- 3. APIデモ実行 (Coreのロジックをシミュレーション) ---

if __name__ == "__main__":
    
    # 1. 初期化
    alice_api = AliceController()
    
    # Coreからの入力値シミュレーション用の初期状態
    current_theta = 0.45      # Full Coreが進化させる
    current_V_total = 0.5     # Full Coreが計算する
    current_R_group = 0.0     # 前回の報酬 (予測に必要)
    
    # Theta進化則のシミュレーション定数 (Full Coreの計算を仮定)
    eta_theta_sim = 1e-5
    gamma_theta_sim = 0.005

    # シミュレーションループ (10ステップ)
    print("\n--- Alice V4.0+ API デモ: Predict-Update サイクル開始 (Core/Controller分離モード) ---")
    
    for step in range(10):
        # 1. Coreからの入力 (K, U)
        K_new = np.random.uniform(0.3, 0.9)
        U_new = np.random.uniform(0.1, 0.8)
        
        # 2. 環境からの報酬フィードバック
        R_feedback_env = 0.9 if step % 2 == 0 else 0.2
        
        print(f"\n[STEP {step+1}] Core Input: K={K_new:.2f}, U={U_new:.2f} | Core State: Theta={current_theta:.4f}, V_total={current_V_total:.4f}")
        print(f"| Environment Feedback R_group = {R_feedback_env:.1f}")

        # 3. Predict: 行動の予測 (Coreの最新状態を全て渡す)
        predicted_action_schema = alice_api.predict_action(
            K_new, U_new, 
            theta_current=current_theta, 
            V_total_current=current_V_total, 
            R_group_current=current_R_group
        )
        
        # 4. Action Output (XR/Robotが実行する部分) - 出力は元のデモと同じ
        emo = predicted_action_schema['internal_state']
        act = predicted_action_schema['embodiment_output']
        
        print(f"  > 感情 E_t (内部状態): Valence={emo['e_valence']:.3f}, Arousal={emo['e_arousal']:.3f}")
        print(f"  > 行動 A_t (外部制御):")
        print(f"    - Gaze Sync Rate: {act['gaze']['sync_rate']:.3f}")
        print(f"    - Facial Intensity: {act['facial_expression']['intensity']:.3f}")
        print(f"    - Body Lean Rad: {act['body_gesture']['lean_angle_rad']:.3f}")
        
        # --- Core Update Logic Simulation ---
        # Coreが R_feedback_env を受け取り V_total と Theta を計算し直すのをシミュレート
        
        # V_total の再計算 (簡易ロジック)
        simulated_Cost_K = K_new * 0.1
        simulated_Cost_U_pz = U_new * 0.05
        simulated_V_total = R_feedback_env - simulated_Cost_K - simulated_Cost_U_pz
        
        # Theta進化シミュレーション (Coreの進化則を簡易的に再現)
        eta_eff_sim = eta_theta_sim * (1.0 - K_new) * (1.0 - U_new)
        eta_eff_sim = max(eta_eff_sim, 1e-7)
        d_theta_sim = eta_eff_sim * (simulated_V_total - 0.5) + gamma_theta_sim * (0.5 - current_theta)
        new_theta = np.clip(current_theta + d_theta_sim, 0.0, 1.0)
        
        # 5. Update: Controllerに Full Core の計算結果を渡して C の学習を実行させる
        updated_state = alice_api.update_state(
            R_group_feedback=R_feedback_env, 
            V_total_canonical=simulated_V_total, 
            K_avg_current=K_new,
            U_pz_current=U_new,
            theta_current=new_theta # Coreで進化した最新のThetaを渡す
        )

        # 6. Internal State After Update: 次のステップのために Core State を更新
        current_theta = updated_state['theta_empathy']
        current_V_total = updated_state['V_total']
        current_R_group = R_feedback_env
        
        print(f"  > 更新後内部状態:")
        print(f"    - Theta (Full Core進化): {updated_state['theta_empathy']:.4f}")
        print(f"    - V_total (Full Core算出): {updated_state['V_total']:.4f}")
        print(f"    - C_G0 (Gaze Gain - Controller学習): {updated_state['C_G0_gaze_gain']:.4f}")
