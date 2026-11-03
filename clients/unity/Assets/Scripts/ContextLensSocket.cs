using System;
using System.Collections.Concurrent;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace ContextLens
{
    // Text-frame WebSocket client. Receiving runs off the main thread; the main thread drains the queues.
    public class ContextLensSocket : IDisposable
    {
        readonly ClientWebSocket ws = new ClientWebSocket();
        readonly CancellationTokenSource cts = new CancellationTokenSource();
        readonly ConcurrentQueue<string> inbox = new ConcurrentQueue<string>();
        readonly ConcurrentQueue<string> events = new ConcurrentQueue<string>();
        readonly SemaphoreSlim sendLock = new SemaphoreSlim(1, 1);

        public bool IsOpen => ws.State == WebSocketState.Open;

        public async Task ConnectAsync(string url)
        {
            await ws.ConnectAsync(new Uri(url), cts.Token);
            _ = ReceiveLoop();
        }

        async Task ReceiveLoop()
        {
            var buffer = new byte[16 * 1024];
            var sb = new StringBuilder();
            try
            {
                while (ws.State == WebSocketState.Open)
                {
                    var r = await ws.ReceiveAsync(new ArraySegment<byte>(buffer), cts.Token);
                    if (r.MessageType == WebSocketMessageType.Close) break;
                    sb.Append(Encoding.UTF8.GetString(buffer, 0, r.Count));
                    if (r.EndOfMessage)
                    {
                        inbox.Enqueue(sb.ToString());
                        sb.Clear();
                    }
                }
            }
            catch (Exception e)
            {
                if (!cts.IsCancellationRequested) events.Enqueue("connection error: " + e.Message);
            }
            events.Enqueue("closed");
        }

        public async void Send<T>(T message)
        {
            if (!IsOpen) return;
            var bytes = Encoding.UTF8.GetBytes(JsonUtility.ToJson(message));
            await sendLock.WaitAsync();
            try
            {
                await ws.SendAsync(new ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, cts.Token);
            }
            catch (Exception e)
            {
                events.Enqueue("send error: " + e.Message);
            }
            finally
            {
                sendLock.Release();
            }
        }

        public bool TryReceive(out string frame) => inbox.TryDequeue(out frame);
        public bool TryEvent(out string ev) => events.TryDequeue(out ev);

        public void Dispose()
        {
            cts.Cancel();
            ws.Dispose();
        }
    }
}
