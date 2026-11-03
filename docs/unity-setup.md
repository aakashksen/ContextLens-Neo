# Unity XR client setup

Project: `clients/unity/` (Unity 6000.3.3f1). It speaks `/ws/{conversation_id}` as described in
`docs/client-contract.md`. Status: the C# was written against the contract and the schema check passes.
**It has not been compiled or run in Unity** (the editor refused to start in batch mode: no valid
license on this machine). Everything under "Manual checks" is for H3.

## Schema check (no Unity needed)

```
python tools/check_unity_schema.py
```

Parses `clients/unity/Assets/Scripts/*.cs` and compares every class against
`docs/client-contract.schema.json`: field names, types, enum values (the `// enum:` comments) and the
`type` constants. Missing, extra or differently typed fields fail with exit code 1.
Standard library only.

## Manual checks (H3)

1. Start the fake server: `python tools/fake_server.py` (needs `pip install -e .[server]`).
2. Open `clients/unity/` in Unity Hub with 6000.3.3f1 (sign in / activate a license). Let it import
   and confirm the Console has no compile errors. Unity will create `Library/`, `ProjectSettings/*`
   and a scene prompt; open or create any empty scene and press Play. The client builds its own
   world-space card at runtime (`ContextLensClient.Boot`), so no scene setup is needed.
3. If the Console complains about the Input System, set Project Settings > Player > Active Input
   Handling to "Both" (the UI uses the legacy event system).
4. Expected on Play: top line "REC: server says recording" (red). If the server is not running it
   reads "Not recording (not connected ...)".
5. Click "Test audio": a transcript line appears. Click "Press" three times: cards come back as
   answered, partial, abstained (see the fake server section of the contract). Pressing before any
   transcript shows "Nothing heard yet" and sends nothing.
6. "Pause" then "Test audio": no new transcript, indicator reads "Not recording (paused)".
   "Resume" restores it.
7. "Delete": transcript and card clear, indicator shows deleted, connection closes.
8. "Webcam": a preview appears and a red "WEBCAM ON (local only, not sent)" line shows; click again to
   turn it off. Nothing from the camera is sent over the socket. (Needs a camera and OS permission.)
9. Controller/simulator: add the XR Interaction Toolkit package, drop its XR Device Simulator into the
   scene, and press the simulated primary button (A/X): it acts like "Press". The client reads
   `CommonUsages.primaryButton` from `UnityEngine.XR.InputDevices`, so it also works with a headset
   controller. Not tested.
10. Untrusted text: card text is shown by a `Text` component with rich text off. Check by
    temporarily making a server card contain `<b>x</b>`: it must show literally.

## Honest limits

- No microphone capture: "Test audio" sends a chunk of silence as a stand-in (the fake server
  ignores audio content). The indicator reflects only what the server last reported, not what this
  client is capturing, because it captures nothing.
- `System.Net.WebSockets.ClientWebSocket` works in the Editor and Windows/Android standalone
  builds, not WebGL.
- JsonUtility cannot represent null, so optional strings (`label`, `suggestion`, `question`,
  `detail`) are empty strings in C#. An empty `question` is sent as `""`, not `null`.
