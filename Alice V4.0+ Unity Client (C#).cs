// ファイル名: AliceWebSocketClient.cs
// Unity C# クライアントスクリプト

using System;
using System.Collections.Generic;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;


// JSON Deserialization Structs (C#でJSONスキーマに対応する構造体)

[Serializable]
public struct GazeData
{
    public float sync_rate;
    public float max_deviation; // Future use
}


[Serializable]
public struct SpeechProsodyData
{
    public float wpm_decay_factor;
    public float pitch_modulation;
}


[Serializable]
public struct FacialExpressionData
{
    public float valence_signal;
    public float intensity;
}


[Serializable]
public struct BodyGestureData
{
    public float assist_priority;
    public float lean_angle_rad;
}


[Serializable]
public struct EmbodimentOutput
{
    public GazeData gaze;
    public SpeechProsodyData speech_prosody;
    public FacialExpressionData facial_expression;
    public BodyGestureData body_gesture;
}


[Serializable]
public struct InternalState
{
    public float theta_empathy;
    public float e_valence;
    public float e_arousal;
}


[Serializable]
public struct AliceActionSchema
{
    public InternalState internal_state;
    public EmbodimentOutput embodiment_output;
}


[Serializable]
public struct AliceResponse
{
    public string status;
    public string action_type;
    public AliceActionSchema data;
}


// クライアント -> サーバーへのリクエスト構造体

[Serializable]
public struct PredictRequest
{
    public string action; // "predict_action"
    public float k_avg;
    public float u_pz;
}


[Serializable]
public struct UpdateRequest
{
    public string action; // "update_state"
    public float r_group;
}


public class AliceWebSocketClient : MonoBehaviour
{
    // WebSocket接続設定
    private const string ServerUri = "ws://127.0.0.1:8765";
    private ClientWebSocket clientWebSocket;
    private CancellationTokenSource cts = new CancellationTokenSource();


    // 内部状態と行動出力の格納
    [Header("Alice V4.0+ Output State")]
    public AliceActionSchema currentActionData;
    [Tooltip("予測アクションの送信間隔 (秒)。例: 0.05秒で20FPS")]
    public float predictionInterval = 0.05f;
    private float timeSinceLastPrediction = 0f;


    // アバター制御用のコンポーネント（ここではAnimatorとSkinnedMeshRendererを想定）
    private Animator animator;
    private SkinnedMeshRenderer meshRenderer;

    // マッピング用の変数
    private float currentLeanAngle = 0f;


    void Start()
    {
        // 必須コンポーネントの取得（アバターにアタッチされている前提）
        animator = GetComponent<Animator>();
        meshRenderer = GetComponentInChildren<SkinnedMeshRenderer>();

        if (animator == null)
        {
            Debug.LogError("Animatorコンポーネントが見つかりません。スクリプトをアバターにアタッチしてください。");
            return;
        }

        // UnityMainThreadDispatcher が利用可能であることを確認するか、適切なメインスレッドディスパッチング機構を用意する必要があります。
        // （元のコードにはその定義がありませんが、Unity環境で非同期処理を行うためには必須のコンポーネントです）
        // WebSocket接続を開始
        ConnectWebSocket();
    }


    private async void ConnectWebSocket()
    {
        clientWebSocket = new ClientWebSocket();
        Debug.Log($"Alice APIサーバーへの接続開始: {ServerUri}");

        try
        {
            await clientWebSocket.ConnectAsync(new Uri(ServerUri), cts.Token);
            Debug.Log("Alice APIサーバーへの接続に成功しました。");

            // 受信ループを別スレッドで開始
            _ = Task.Run(() => ReceiveLoop());
        }
        catch (Exception e)
        {
            Debug.LogError($"WebSocket接続エラー: {e.Message}");
        }
    }


    private async Task ReceiveLoop()
    {
        byte[] buffer = new byte[1024 * 4];
        while (clientWebSocket.State == WebSocketState.Open && !cts.IsCancellationRequested)
        {
            try
            {
                var result = await clientWebSocket.ReceiveAsync(new ArraySegment<byte>(buffer), cts.Token);

                if (result.MessageType == WebSocketMessageType.Close)
                {
                    await clientWebSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, string.Empty, CancellationToken.None);
                    Debug.Log("サーバーが切断しました。");
                    break;
                }

                string responseString = Encoding.UTF8.GetString(buffer, 0, result.Count);

                // UnityのメインスレッドでJSON処理とアバター更新を実行
                // UnityMainThreadDispatcher がない場合は、以下を修正する必要があります
                // 例: UnityMainThreadDispatcher.Instance().Enqueue(() => ProcessActionResponse(responseString));
                // 暫定的に非同期に処理
                // TODO: ここをメインスレッド処理に置き換える
                ProcessActionResponse(responseString);
            }
            catch (Exception e)
            {
                if (e is TaskCanceledException || e is OperationCanceledException)
                {
                    // 意図的な切断
                }
                else
                {
                    Debug.LogError($"WebSocket受信エラー: {e.Message}");
                }
                break;
            }
        }
    }


    private void ProcessActionResponse(string responseJson)
    {
        // JSONResponse全体をデコード
        AliceResponse response = JsonUtility.FromJson<AliceResponse>(responseJson);

        if (response.status == "success" && response.action_type == "predicted_action")
        {
            // predicted_action のデータのみを処理
            currentActionData = response.data;
            ApplyEmbodimentToAvatar();
        }
        else if (response.status == "success" && response.action_type == "updated_state")
        {
            // update_state の結果をデバッグ表示（学習結果）
            Debug.Log($"[Alice Update] Theta:{response.data.internal_state.theta_empathy:F4} | V_Total:{response.data.internal_state.e_valence:F4}");
        }
    }

    // アバターに身体化写像を適用するメイン関数
    private void ApplyEmbodimentToAvatar()
    {
        // 1. 表情 (Facial Expression) の適用
        // Unityのブレンドシェイプ名はプロジェクト依存。ここでは仮の名称を使用。
        if (meshRenderer != null)
        {
            float v = currentActionData.internal_state.e_valence;
            float intensity = currentActionData.embodiment_output.facial_expression.intensity;

            // Valenceに基づいて表情をブレンド（例：Positive / Negative）
            if (v >= 0)
            {
                // Positive (Joy/Interest)
                // SetBlendShapeWeight(インデックス, 重み0-100)
                meshRenderer.SetBlendShapeWeight(0, intensity * v * 100); 
                meshRenderer.SetBlendShapeWeight(1, 0);
            }
            else
            {
                // Negative (Stress/Discomfort)
                meshRenderer.SetBlendShapeWeight(0, 0);
                meshRenderer.SetBlendShapeWeight(1, intensity * Mathf.Abs(v) * 100);
            }
            // Arousalに基づいた目の開きや眉のブレンドシェイプも制御可能
        }


        // 2. 視線 (Gaze) の適用
        // LookAt IKのウェイトを制御（Gaze Sync Rateに依存）
        // Animator.SetLookAtWeight(currentActionData.embodiment_output.gaze.sync_rate);
        // 実際のUnity IKシステムに合わせて実装が必要


        // 3. 姿勢・傾き (Body Lean) の適用
        float targetLeanRad = currentActionData.embodiment_output.body_gesture.lean_angle_rad;
        // スムージング処理
        currentLeanAngle = Mathf.Lerp(currentLeanAngle, targetLeanRad, Time.deltaTime * 5f);

        // AnimatorのFloatパラメータにマッピング（Animator Controller側で処理が必要）
        animator.SetFloat("BodyLeanAngle", currentLeanAngle * Mathf.Rad2Deg);

        // 4. 音声プロソディ (Speech Prosody) への適用
        // TTS (Text-to-Speech) エンジンに渡すパラメーター
        float pitch = 1.0f + currentActionData.embodiment_output.speech_prosody.pitch_modulation * 0.5f; // 1.0 ~ 1.5倍
        float speed = 1.0f - currentActionData.embodiment_output.speech_prosody.wpm_decay_factor * 0.2f; // 0.8 ~ 1.0倍

        // ここで Unity TTS Engine や Audio Source に pitch, speed を適用
        // Debug.Log($"TTS Params: Pitch={pitch:F2}, Speed={speed:F2}");
    }


    // Updateはクライアントからの予測を定期的にトリガーするために使用
    void Update()
    {
        timeSinceLastPrediction += Time.deltaTime;

        if (clientWebSocket != null && clientWebSocket.State == WebSocketState.Open && timeSinceLastPrediction >= predictionInterval)
        {
            timeSinceLastPrediction = 0f;

            // 外部環境から取得したK_avgとU_pzを入力として送信
            // 仮のダミー入力を使用。実際はゲーム内のCognitive modelから取得。
            float dummy_K = Mathf.Sin(Time.time * 0.5f) * 0.2f + 0.6f;
            float dummy_U = Mathf.Cos(Time.time * 0.3f) * 0.15f + 0.4f;

            SendPredictAction(dummy_K, dummy_U);

            // 例: 5秒に一度、高報酬フィードバックを送信（学習のトリガー）
            if (Time.frameCount % (int)(5.0f / predictionInterval) == 0)
            {
                SendUpdateState(UnityEngine.Random.Range(0.7f, 0.9f));
            }
        }
    }


    // Predict Action リクエストをサーバーへ送信
    public void SendPredictAction(float k_avg, float u_pz)
    {
        PredictRequest request = new PredictRequest
        {
            action = "predict_action",
            k_avg = k_avg,
            u_pz = u_pz
        };
        string json = JsonUtility.ToJson(request);
        SendWebSocketMessage(json);
    }


    // Update State リクエストをサーバーへ送信
    public void SendUpdateState(float r_group)
    {
        UpdateRequest request = new UpdateRequest
        {
            action = "update_state",
            r_group = r_group
        };
        string json = JsonUtility.ToJson(request);
        SendWebSocketMessage(json);
    }


    private async void SendWebSocketMessage(string message)
    {
        if (clientWebSocket != null && clientWebSocket.State == WebSocketState.Open)
        {
            byte[] buffer = Encoding.UTF8.GetBytes(message);
            try
            {
                await clientWebSocket.SendAsync(new ArraySegment<byte>(buffer), WebSocketMessageType.Text, true, cts.Token);
            }
            catch (Exception e)
            {
                Debug.LogError($"送信エラー: {e.Message}");
            }
        }
    }


    void OnDestroy()
    {
        // アプリケーション終了時に接続を閉じる
        cts.Cancel();
        if (clientWebSocket != null)
        {
            clientWebSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Client closing", CancellationToken.None).Wait();
            clientWebSocket.Dispose();
        }
    }
}
