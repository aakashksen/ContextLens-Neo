using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.UI;

namespace ContextLens
{
    // Builds a small world-space card UI at runtime and drives it from the /ws/{conversation_id} contract.
    public class ContextLensClient : MonoBehaviour
    {
        public string serverUrl = "ws://127.0.0.1:8000/ws/";
        public string conversationId = "demo";
        // "Test audio" sends a canned clip to check the setup (docs/unity-setup.md). Off for a clean wearer view.
        public bool showDevControls = true;

        ContextLensSocket socket;
        readonly List<Turn> turns = new List<Turn>();
        // What the indicator may claim. Same rules as clients/web/src/session.ts: the user's own Pause,
        // Resume and Delete presses take effect here at once, and a "recording" status the user did
        // not ask for (after Pause, Delete or a closed socket) is stale and ignored.
        string connection = "connecting"; // connecting | recording | paused | deleting | deleted | closed
        bool resumePending; // from the user's Resume press until the server answers recording
        string connectionDetail = "";
        WebCamTexture webcam;
        Text cardText, stateText, webcamText, transcriptText;
        Text cardHead;
        RawImage webcamPreview;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Boot()
        {
            if (FindFirstObjectByType<ContextLensClient>() != null) return;
            new GameObject("ContextLensClient").AddComponent<ContextLensClient>();
        }

        async void Start()
        {
            BuildUi();
            socket = new ContextLensSocket();
            try
            {
                await socket.ConnectAsync(serverUrl + conversationId);
            }
            catch (Exception e)
            {
                connection = "closed";
                connectionDetail = e.Message;
            }
            RefreshState();
        }

        void Update()
        {
            if (socket == null) return;
            while (socket.TryReceive(out var frame)) Handle(frame);
            while (socket.TryEvent(out var ev))
            {
                if (ev == "closed")
                {
                    resumePending = false;
                    if (connection != "deleted") connection = "closed";
                }
                else connectionDetail = ev;
                RefreshState();
            }
        }

        void Handle(string frame)
        {
            // Frames are untrusted data: a parse failure is shown, never acted on.
            try
            {
                switch (JsonUtility.FromJson<Envelope>(frame).type)
                {
                    case "transcript":
                        turns.Add(JsonUtility.FromJson<TranscriptMessage>(frame).turn);
                        RefreshTranscript();
                        break;
                    case "turn_update":
                        // The same turn with its speaker now known: replace it, never append.
                        var update = JsonUtility.FromJson<TurnUpdateMessage>(frame).turn;
                        var at = turns.FindIndex(t => t.id == update.id);
                        if (at >= 0)
                        {
                            turns[at] = update;
                            RefreshTranscript();
                        }
                        break;
                    case "card":
                        ShowCard(JsonUtility.FromJson<CardMessage>(frame).card);
                        break;
                    case "status":
                        var s = JsonUtility.FromJson<StatusMessage>(frame);
                        if (s.state == "error")
                        {
                            cardText.text = "Server error: " + s.detail;
                        }
                        else if (s.state == "deleted")
                        {
                            resumePending = false;
                            connection = "deleted";
                            turns.Clear();
                            cardText.text = "Transcript deleted.";
                            RefreshTranscript();
                        }
                        else
                        {
                            ReceiveStatus(s.state);
                        }
                        RefreshState();
                        break;
                    default:
                        cardText.text = "Unknown message type ignored.";
                        break;
                }
            }
            catch (Exception e)
            {
                cardText.text = "Could not read a server message: " + e.Message;
            }
        }

        // Capture may start only from the first connect or a Resume the user pressed.
        void ReceiveStatus(string state)
        {
            if (connection == "deleting" || connection == "deleted" || connection == "closed") return;
            if (state == "recording")
            {
                if (connection == "paused" && !resumePending) return;
                resumePending = false;
                connection = "recording";
            }
            else if (state == "paused")
            {
                connection = "paused";
            }
        }

        // Called by a button, and by ContextLensXRInput when the controller or simulator primary button goes down.
        public void Press()
        {
            if (turns.Count == 0)
            {
                cardText.text = "Nothing heard yet. Send test audio first.";
                return;
            }
            // as_of_seq is the latest transcript turn this client holds, as the contract asks.
            socket.Send(new PressMessage
            {
                request = new ButtonRequest
                {
                    conversation_id = conversationId,
                    as_of_seq = turns[turns.Count - 1].seq,
                    detail = "standard"
                }
            });
        }

        // A chunk of silence: this client does not capture the microphone.
        // Sent only while this client itself believes it is recording.
        public void SendTestAudio()
        {
            if (connection != "recording") return;
            socket.Send(new AudioMessage { data = Convert.ToBase64String(new byte[3200]) });
        }

        public void Pause()
        {
            if (connection != "recording") return;
            resumePending = false;
            connection = "paused"; // shown at once, whatever the server later says
            RefreshState();
            socket.Send(new PauseMessage());
        }

        public void Resume()
        {
            if (connection != "paused" || resumePending) return;
            resumePending = true;
            socket.Send(new ResumeMessage());
            RefreshState();
        }

        public void DeleteTranscript()
        {
            if (connection == "deleting" || connection == "deleted" || connection == "closed") return;
            resumePending = false;
            connection = "deleting";
            RefreshState();
            socket.Send(new DeleteMessage());
        }

        public void ToggleWebcam()
        {
            if (webcam != null && webcam.isPlaying)
            {
                webcam.Stop();
                webcamPreview.texture = null;
                webcamPreview.gameObject.SetActive(false);
            }
            else if (WebCamTexture.devices.Length > 0)
            {
                // Frames stay in this process: they are only drawn to the preview, never sent.
                webcam = new WebCamTexture(WebCamTexture.devices[0].name);
                webcam.Play();
                webcamPreview.texture = webcam;
                webcamPreview.gameObject.SetActive(true);
            }
            bool on = webcam != null && webcam.isPlaying;
            webcamText.text = on ? "WEBCAM ON (local only, not sent)" : "";
            webcamText.color = on ? Color.red : Color.gray;
        }

        void ShowCard(Card c)
        {
            var sb = new StringBuilder();
            cardHead.text = StatusLabel(c.status);
            cardHead.color = c.status == "answered" ? Ok : c.status == "partial" ? Accent : Color.gray;
            foreach (var f in c.findings)
            {
                sb.AppendLine("\u2022 " + f.statement);
                foreach (var e in f.evidence)
                    sb.AppendLine("     " + e.speaker + ", " + Clock(e.start_ms) + ":  \u201c" + e.quote + "\u201d");
            }
            if (!string.IsNullOrEmpty(c.suggestion)) sb.AppendLine("\nNext: " + c.suggestion);
            if (c.missing != null && c.missing.Count > 0) sb.AppendLine("Not said yet: " + string.Join("; ", c.missing));
            cardText.text = sb.ToString();
        }

        static readonly Color Accent = new Color(0.98f, 0.75f, 0.25f);
        static readonly Color Ok = new Color(0.45f, 0.85f, 0.55f);

        static string StatusLabel(string status) =>
            status == "answered" ? "Answered" : status == "partial" ? "Partly answered" : status == "abstained" ? "Not discussed yet" : status;

        static string Clock(long ms)
        {
            long s = Math.Max(0, ms / 1000);
            return (s / 60) + ":" + (s % 60).ToString("00");
        }

        void RefreshTranscript()
        {
            var sb = new StringBuilder();
            for (int i = Math.Max(0, turns.Count - 4); i < turns.Count; i++)
                sb.AppendLine(turns[i].speaker + ": " + turns[i].text);
            transcriptText.text = sb.ToString();
        }

        void RefreshState()
        {
            // "REC" appears only when the server has confirmed recording and the user has not pressed
            // Pause or Delete since. Resume stays "paused" until the server answers.
            bool rec = connection == "recording";
            string label = connection == "closed" ? "not connected" + (connectionDetail != "" ? " (" + connectionDetail + ")" : "") : connection;
            stateText.text = rec ? "\u25cf REC  listening" : "Not recording (" + label + ")";
            stateText.color = rec ? Color.red : Color.gray;
        }

        void BuildUi()
        {
            var canvasGo = new GameObject("Card", typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
            canvasGo.GetComponent<Canvas>().renderMode = RenderMode.WorldSpace;
            var rt = canvasGo.GetComponent<RectTransform>();
            rt.sizeDelta = new Vector2(800, 600);
            rt.position = new Vector3(0, 1.5f, 2f);
            rt.localScale = Vector3.one * 0.0015f;
            var bg = new GameObject("Bg", typeof(Image)).GetComponent<Image>();
            bg.transform.SetParent(rt, false);
            bg.color = new Color(0.07f, 0.08f, 0.11f, 0.92f);
            bg.rectTransform.sizeDelta = rt.sizeDelta;

            var bar = new GameObject("Accent", typeof(Image)).GetComponent<Image>();
            bar.transform.SetParent(rt, false);
            bar.color = Accent;
            bar.rectTransform.sizeDelta = new Vector2(rt.sizeDelta.x, 6);
            bar.rectTransform.anchoredPosition = new Vector2(0, rt.sizeDelta.y / 2 - 3);

            stateText = MakeText(rt, new Vector2(0, 268), new Vector2(760, 36), 24);
            webcamText = MakeText(rt, new Vector2(0, 268), new Vector2(760, 36), 24);
            webcamText.alignment = TextAnchor.UpperRight;
            webcamText.color = Color.gray;
            transcriptText = MakeText(rt, new Vector2(0, 160), new Vector2(760, 130), 22);
            transcriptText.color = new Color(0.7f, 0.72f, 0.78f);
            cardHead = MakeText(rt, new Vector2(0, 70), new Vector2(760, 44), 32);
            cardHead.fontStyle = FontStyle.Bold;
            cardText = MakeText(rt, new Vector2(0, -75), new Vector2(760, 245), 26);
            cardText.text = "Press to ask what was said.";

            webcamPreview = new GameObject("WebcamPreview", typeof(RawImage)).GetComponent<RawImage>();
            webcamPreview.transform.SetParent(rt, false);
            webcamPreview.rectTransform.sizeDelta = new Vector2(160, 120);
            webcamPreview.rectTransform.anchoredPosition = new Vector2(300, 190);
            webcamPreview.gameObject.SetActive(false);

            // Press leads; Pause, Resume and Delete stay one tap away (privacy controls).
            MakeButton(rt, "Press", -290, -250, Press, 160, Accent, new Color(0.12f, 0.1f, 0.05f));
            MakeButton(rt, "Pause", -130, -250, Pause);
            MakeButton(rt, "Resume", 10, -250, Resume);
            MakeButton(rt, "Webcam", 150, -250, ToggleWebcam);
            MakeButton(rt, "Delete", 290, -250, DeleteTranscript);
            if (showDevControls) MakeButton(rt, "Test audio", 300, 105, SendTestAudio);

            if (FindFirstObjectByType<UnityEngine.EventSystems.EventSystem>() == null)
                new GameObject("EventSystem", typeof(UnityEngine.EventSystems.EventSystem),
                    typeof(UnityEngine.EventSystems.StandaloneInputModule));
            RefreshState();
        }

        static Text MakeText(Transform parent, Vector2 pos, Vector2 size, int fontSize)
        {
            var go = new GameObject("Text", typeof(Text));
            go.transform.SetParent(parent, false);
            var t = go.GetComponent<Text>();
            t.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            t.fontSize = fontSize;
            t.supportRichText = false; // card text is untrusted: no markup parsing
            t.color = Color.white;
            t.alignment = TextAnchor.UpperLeft;
            t.horizontalOverflow = HorizontalWrapMode.Wrap;
            t.verticalOverflow = VerticalWrapMode.Truncate;
            t.rectTransform.sizeDelta = size;
            t.rectTransform.anchoredPosition = pos;
            return t;
        }

        static void MakeButton(Transform parent, string label, float x, float y,
            UnityEngine.Events.UnityAction onClick, float width = 130, Color? fill = null, Color? ink = null)
        {
            var go = new GameObject(label + "Button", typeof(Image), typeof(Button));
            go.transform.SetParent(parent, false);
            go.GetComponent<Image>().color = fill ?? new Color(0.2f, 0.23f, 0.3f);
            var rt = go.GetComponent<RectTransform>();
            rt.sizeDelta = new Vector2(width, 50);
            rt.anchoredPosition = new Vector2(x, y);
            go.GetComponent<Button>().onClick.AddListener(onClick);
            var t = MakeText(rt, Vector2.zero, new Vector2(width, 50), 22);
            t.alignment = TextAnchor.MiddleCenter;
            t.color = ink ?? Color.white;
            t.text = label;
        }

        void OnDestroy()
        {
            if (webcam != null) webcam.Stop();
            socket?.Dispose();
        }
    }
}
