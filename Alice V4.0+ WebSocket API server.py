# -*- coding: utf-8 -*-

import asyncio
import json
import websockets
import time
import random
import sys

# 前回作成した AliceController をインポート
# 注: このファイルを実行するには、alice_v4_plus_api_controller.py が必要です
try:
    from alice_v4_plus_api_controller import AliceController
except ImportError:
    print("エラー: alice_v4_plus_api_controller.py が見つかりません。")
    print("WebSocketサーバーを起動する前に、前回のファイルを同じディレクトリに配置してください。")
    sys.exit(1)


# --- 1. グローバル設定とインスタンス化 ---

HOST = "127.0.0.1"
PORT = 8765
alice_controller = AliceController()  # Alice V4.0+ コントローラーを初期化


# --- 2. WebSocket ハンドラー関数 ---

async def alice_websocket_handler(websocket, path):
    """
    WebSocketからのリクエスト（JSON）を受け取り、AliceControllerを介して処理し、
    結果をJSONで返す非同期ハンドラー。
    """
    print(f"[{time.strftime('%H:%M:%S')}] クライアント接続: {websocket.remote_address}")

    try:
        async for message in websocket:
            # JSONメッセージを解析
            try:
                data = json.loads(message)
                action_type = data.get("action", "").lower()
            except json.JSONDecodeError:
                error_response = {"status": "error", "message": "無効なJSON形式です"}
                await websocket.send(json.dumps(error_response))
                continue

            # リクエストタイプに応じて処理を分岐
            if action_type == "predict_action":
                # === 予測処理（高速処理が要求される）===
                # 外部環境（XR/Robot）からの入力：K_avg, U_pz
                k_avg_new = data.get("k_avg", 0.5)
                u_pz_new = data.get("u_pz", 0.5)

                # Alice Controller で行動を予測
                predicted_action_schema = alice_controller.predict_action(k_avg_new, u_pz_new)

                # 結果をJSONで返信 (Action Schema)
                response = {
                    "status": "success",
                    "action_type": "predicted_action",
                    "timestamp": time.time(),
                    "data": predicted_action_schema
                }

            elif action_type == "update_state":
                # === 更新処理（学習勾配計算が含まれる）===
                # 外部環境（XR/Robot）からのフィードバック：R_group
                r_group_feedback = data.get("r_group", 0.5)

                # Alice Controller で状態を更新（二層学習が実行される）
                updated_state = alice_controller.update_state(r_group_feedback)

                # 結果をJSONで返信
                response = {
                    "status": "success",
                    "action_type": "updated_state",
                    "timestamp": time.time(),
                    "data": updated_state
                }

            elif action_type == "get_state":
                # === 状態取得（デバッグ/モニタリング用）===
                current_state = alice_controller.get_current_state()
                response = {
                    "status": "success",
                    "action_type": "current_state",
                    "timestamp": time.time(),
                    "data": current_state
                }

            else:
                response = {"status": "error", "message": f"不明なアクションタイプ: {action_type}"}

            # クライアントへレスポンスを送信
            await websocket.send(json.dumps(response))

    except websockets.exceptions.ConnectionClosedOK:
        print(f"[{time.strftime('%H:%M:%S')}] クライアント切断: {websocket.remote_address}")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] エラー発生: {e}")
        error_response = {"status": "error", "message": str(e)}
        try:
            await websocket.send(json.dumps(error_response))
        except:
            pass  # 切断済みの場合の更なるエラーを回避


# --- 3. サーバー起動ロジック ---

async def start_server():
    """
    WebSocketサーバーを起動するメイン関数。
    """
    # 接続テスト用のサンプルクライアントデータ
    print("\n--- サンプルクライアント リクエスト構造（Unity/Unrealが送信） ---")
    print("Predict Request (高速・高頻度):")
    print(json.dumps({"action": "predict_action", "k_avg": 0.7, "u_pz": 0.3}))
    print("\nUpdate Request (低速・不定期):")
    print(json.dumps({"action": "update_state", "r_group": 0.9}))
    print("-" * 60)

    # WebSocketサーバーを起動
    async with websockets.serve(alice_websocket_handler, HOST, PORT):
        print(f"[{time.strftime('%H:%M:%S')}] Alice V4.0+ WebSocket API サーバー起動中...")
        print(f"ホスト: ws://{HOST}:{PORT}")
        print("クライアントからの接続を待機しています...")
        # サーバーが永続的に実行されるようにブロック
        await asyncio.Future()


if __name__ == "__main__":
    try:
        # Python 3.7+ の asyncio を使用してサーバーを起動
        asyncio.run(start_server())
    except KeyboardInterrupt:
        print(f"\n[{time.strftime('%H:%M:%S')}] サーバーをシャットダウンしています...")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] 致命的なエラー: {e}")
