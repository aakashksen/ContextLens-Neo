#if UNITY_EDITOR
using System;
using System.IO;
using System.Reflection;
using ContextLens;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

// Renders the client's own world-space card, fed real wire frames, to PNGs.
// Run: Unity -batchmode -projectPath clients/unity -executeMethod CardCapture.Run -outDir <dir>
public static class CardCapture
{
    const string Turns =
        "{\"type\":\"transcript\",\"turn\":{\"id\":\"d-t%N%\",\"conversation_id\":\"demo\",\"seq\":%N%,\"speaker\":\"%S%\",\"speaker_confidence\":1,\"text\":\"%T%\",\"start_ms\":%M%,\"end_ms\":%E%,\"source\":\"synthetic\"}}";

    const string CardFrame =
        "{\"type\":\"card\",\"card\":{\"mode\":\"recall\",\"status\":\"answered\",\"findings\":[" +
        "{\"agent\":\"recall\",\"kind\":\"recall\",\"statement\":\"The beta moved from Friday to Monday.\",\"evidence\":[{\"turn_id\":\"d-t3\",\"quote\":\"Let us push the beta to Monday.\",\"speaker\":\"Maya\",\"start_ms\":12000}],\"label\":\"update\",\"confidence\":0.92}," +
        "{\"agent\":\"consistency\",\"kind\":\"consistency\",\"statement\":\"Changes what was said earlier: Maya first said Friday.\",\"evidence\":[{\"turn_id\":\"d-t0\",\"quote\":\"The beta ships Friday.\",\"speaker\":\"Maya\",\"start_ms\":0}],\"label\":\"update\",\"confidence\":0.9}]," +
        "\"suggestion\":\"Tell Sam the new date.\",\"missing\":[],\"latency_ms\":1200,\"tokens\":350,\"cost_usd\":0.002}}";

    static void Set(object o, string name, object v) =>
        o.GetType().GetField(name, BindingFlags.NonPublic | BindingFlags.Instance).SetValue(o, v);
    static void Call(object o, string name, params object[] a) =>
        o.GetType().GetMethod(name, BindingFlags.NonPublic | BindingFlags.Instance).Invoke(o, a);

    static Material Mat(Color c, float smooth = 0.2f)
    {
        var m = new Material(Shader.Find("Standard")) { color = c };
        m.SetFloat("_Glossiness", smooth);
        return m;
    }

    static GameObject Prim(PrimitiveType t, Vector3 p, Vector3 s, Color c, Vector3? rot = null)
    {
        var g = GameObject.CreatePrimitive(t);
        g.transform.position = p; g.transform.localScale = s;
        if (rot.HasValue) g.transform.eulerAngles = rot.Value;
        g.GetComponent<Renderer>().sharedMaterial = Mat(c);
        return g;
    }

    public static void Run()
    {
        string outDir = "shots";
        var args = Environment.GetCommandLineArgs();
        for (int i = 0; i < args.Length - 1; i++) if (args[i] == "-outDir") outDir = args[i + 1];
        Directory.CreateDirectory(outDir);
        try { Render(outDir); EditorApplication.Exit(0); }
        catch (Exception e) { Debug.LogError(e); EditorApplication.Exit(1); }
    }

    static void Render(string outDir)
    {
        EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

        // Dim meeting-room: floor, back wall, table, chairs, window glow.
        Prim(PrimitiveType.Cube, new Vector3(0, -0.05f, 4), new Vector3(10, 0.1f, 10), new Color(0.10f, 0.10f, 0.12f));
        Prim(PrimitiveType.Cube, new Vector3(0, 1.8f, 7.5f), new Vector3(10, 4, 0.2f), new Color(0.16f, 0.17f, 0.2f));
        Prim(PrimitiveType.Cube, new Vector3(-2.2f, 1.9f, 7.35f), new Vector3(2.6f, 1.6f, 0.05f), new Color(0.35f, 0.45f, 0.6f));
        Prim(PrimitiveType.Cube, new Vector3(0, 0.72f, 4.8f), new Vector3(3.2f, 0.06f, 1.4f), new Color(0.22f, 0.15f, 0.1f));
        foreach (var x in new[] { -1.4f, 1.4f })
            foreach (var z in new[] { 4.3f, 5.3f })
                Prim(PrimitiveType.Cube, new Vector3(x, 0.36f, z), new Vector3(0.06f, 0.72f, 0.06f), new Color(0.08f, 0.08f, 0.08f));
        for (int i = -2; i <= 2; i++)
            Prim(PrimitiveType.Cube, new Vector3(i * 0.62f, 0.45f, 5.75f), new Vector3(0.45f, 0.08f, 0.45f), new Color(0.12f, 0.14f, 0.2f));
        for (int i = -2; i <= 2; i++)
            Prim(PrimitiveType.Cube, new Vector3(i * 0.62f, 0.8f, 6.0f), new Vector3(0.45f, 0.6f, 0.06f), new Color(0.12f, 0.14f, 0.2f));
        Prim(PrimitiveType.Sphere, new Vector3(0.6f, 0.82f, 4.7f), new Vector3(0.14f, 0.14f, 0.14f), new Color(0.8f, 0.8f, 0.85f));
        Prim(PrimitiveType.Cylinder, new Vector3(-0.8f, 0.8f, 4.9f), new Vector3(0.09f, 0.07f, 0.09f), new Color(0.6f, 0.2f, 0.2f));

        var sun = new GameObject("Light").AddComponent<Light>();
        sun.type = LightType.Directional; sun.intensity = 0.55f; sun.color = new Color(0.7f, 0.78f, 1f);
        sun.transform.eulerAngles = new Vector3(55, -25, 0);
        var pl = new GameObject("Lamp").AddComponent<Light>();
        pl.type = LightType.Point; pl.range = 8; pl.intensity = 1.2f; pl.color = new Color(1f, 0.85f, 0.6f);
        pl.transform.position = new Vector3(0, 2.8f, 4.5f);
        RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Flat;
        RenderSettings.ambientLight = new Color(0.07f, 0.08f, 0.11f);

        var camGo = new GameObject("Cam");
        var cam = camGo.AddComponent<Camera>();
        cam.clearFlags = CameraClearFlags.SolidColor; cam.backgroundColor = new Color(0.03f, 0.035f, 0.05f);
        cam.fieldOfView = 55; cam.nearClipPlane = 0.05f; cam.farClipPlane = 50;
        camGo.transform.position = new Vector3(0, 1.5f, 0.45f);
        camGo.transform.eulerAngles = new Vector3(2, 0, 0);

        // The client, driven through its own private build and frame handling.
        var client = new GameObject("ContextLensClient").AddComponent<ContextLensClient>();
        client.showDevControls = false; // the wearer's view, without the setup check button
        Call(client, "BuildUi");
        string[,] lines = {
            { "Maya", "The beta ships Friday." },
            { "Jordan", "QA still needs the staging build." },
            { "Priya", "Docs will be ready by Thursday." },
            { "Maya", "Let us push the beta to Monday." } };
        for (int i = 0; i < 4; i++)
            Call(client, "Handle", Turns.Replace("%M%", (i * 4000).ToString()).Replace("%E%", (i * 4000 + 3500).ToString()).Replace("%N%", i.ToString()).Replace("%S%", lines[i, 0]).Replace("%T%", lines[i, 1]));
        Call(client, "Handle", CardFrame);
        Set(client, "connection", "recording");
        Call(client, "RefreshState");

        Canvas.ForceUpdateCanvases();
        foreach (var c in UnityEngine.Object.FindObjectsByType<Canvas>(FindObjectsSortMode.None)) c.worldCamera = cam;

        Shot(cam, 1600, 1000, Path.Combine(outDir, "glasses.png"));
        Shot(cam, 1200, 900, Path.Combine(outDir, "glasses-1200x900.png"));
    }

    static void Shot(Camera cam, int w, int h, string path)
    {
        var rt = new RenderTexture(w, h, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB) { antiAliasing = 4 };
        cam.targetTexture = rt;
        Canvas.ForceUpdateCanvases();
        cam.Render(); cam.Render();
        RenderTexture.active = rt;
        var tex = new Texture2D(w, h, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, w, h), 0, 0); tex.Apply();
        File.WriteAllBytes(path, tex.EncodeToPNG());
        RenderTexture.active = null; cam.targetTexture = null;
        Debug.Log("CardCapture wrote " + path);
    }
}
#endif
