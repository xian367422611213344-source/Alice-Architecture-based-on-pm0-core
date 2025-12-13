// ファイル名: AliceDemoUIManager.cs
// Unity デモUI管理スクリプト

using UnityEngine;
using UnityEngine.UI;
using System.Globalization;

public class AliceDemoUIManager : MonoBehaviour
{
    [Header("Dependencies")]
    [Tooltip("WebSocketクライアントコンポーネントをアタッチ")]
    public AliceWebSocketClient client;


    [Header("Input Sliders (Environment Simulation)")]
    public Slider kAvgSlider;
    public Slider uPzSlider;


    [Header("Update Feedback")]
    public Slider rGroupSlider;
    public Button updateButton;


    [Header("Monitoring Text Fields")]
    public Text kAvgText;
    public Text uPzText;
    public Text rGroupText;
    public Text thetaText;
    public Text valenceText;
    public Text arousalText;


    private float currentKAvg = 0.5f;
    private float currentUPz = 0.5f;
    private float currentRGroup = 0.5f;


    void Start()
    {
        // クライアントの存在確認
        if (client == null)
        {
            client = FindObjectOfType<AliceWebSocketClient>();
            if (client == null)
            {
                Debug.LogError("AliceWebSocketClient がシーンに見つかりません。コンポーネントが存在するか確認してください。");
                return;
            }
        }

        // UIイベントリスナーの設定
        kAvgSlider.onValueChanged.AddListener(OnKAvgChanged);
        uPzSlider.onValueChanged.AddListener(OnUPzChanged);
        rGroupSlider.onValueChanged.AddListener(OnRGroupChanged);
        updateButton.onClick.AddListener(OnUpdateButtonClicked);

        // 初期値を設定
        OnKAvgChanged(kAvgSlider.value);
        OnUPzChanged(uPzSlider.value);
        OnRGroupChanged(rGroupSlider.value);
    }


    // K_avg (認知負荷) スライダーが変更されたとき
    private void OnKAvgChanged(float value)
    {
        currentKAvg = value;
        kAvgText.text = $"K_avg (認知負荷): {value:F2}";
        // K_avgが変更されたら即座に予測リクエストを送信
        UpdatePrediction();
    }


    // U_pz (予測不安) スライダーが変更されたとき
    private void OnUPzChanged(float value)
    {
        currentUPz = value;
        uPzText.text = $"U_pz (予測不安): {value:F2}";
        // U_pzが変更されたら即座に予測リクエストを送信
        UpdatePrediction();
    }


    // R_group (報酬フィードバック) スライダーが変更されたとき
    private void OnRGroupChanged(float value)
    {
        currentRGroup = value;
        rGroupText.text = $"R_group (報酬): {value:F2}";
    }


    // Predict Action をクライアント経由でサーバーへ送信
    private void UpdatePrediction()
    {
        if (client != null)
        {
            // リアルタイムでKとUの値をサーバーに送り、行動予測を要求する
            client.SendPredictAction(currentKAvg, currentUPz);
        }
    }


    // Update State ボタンがクリックされたとき
    private void OnUpdateButtonClicked()
    {
        if (client != null)
        {
            // Update State (学習) リクエストを送信
            client.SendUpdateState(currentRGroup);
            Debug.Log($"[UI Event] Update State を実行: R_group={currentRGroup:F2}");
        }
    }


    void Update()
    {
        // 予測リクエストの結果をUIにリアルタイムで反映
        if (client != null)
        {
            // 内部状態のモニタリング
            thetaText.text = $"Theta (人格): {client.currentActionData.internal_state.theta_empathy:F4}";
            valenceText.text = $"Valence (快不快): {client.currentActionData.internal_state.e_valence:F4}";
            arousalText.text = $"Arousal (覚醒): {client.currentActionData.internal_state.e_arousal:F4}";

            // 行動出力のモニタリング (必要に応じて追加)
            // 例: gazeText.text = $"Gaze: {client.currentActionData.embodiment_output.gaze.sync_rate:F2}";
        }
    }
}
