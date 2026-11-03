# Unity XR client setup

Project: `clients/unity/` (Unity 6000.3.3f1). It speaks `/ws/{conversation_id}` as described in
`docs/client-contract.md`. Status: the C# was written against the contract and the schema check passes.
**It has not been run in Unity** (the editor refused to start in batch mode: no valid license on this
machine). It was compiled outside the Editor with the Roslyn compiler bundled in the Unity install, against
the installed `UnityEngine*.dll` modules and the Unity-shipped `UnityEngine.UI.dll` and `Unity.InputSystem.dll`
(template cache). That proves the C# is valid against those APIs, not that the Editor accepts the project or
that anything works at runtime. Everything under "Manual checks" is for H3.

## Schema check (no Unity needed)

```
python tools/check_unity_schema.py
```

Parses `clients/unity/Assets/Scripts/Messages.cs` and compares every class against
`docs/client-contract.schema.json`: field names, types, enum values (the `// enum:` comments), `// nullable`
markers for fields that are null on the wire, and defaults (including the `type` constants; required
fields must have none). It fails closed: any line in `Messages.cs` it cannot read (attributes on a field,
private or `[SerializeField]` fields, properties, two fields on a line, one-line classes) is reported as a
problem, and `[Serializable]` types outside `Messages.cs` are refused. Exit code 1 on any problem.
Standard library only. It checks the message types only, not the client logic.

## Manual checks (H3)

1. Start the fake server: `python tools/fake_server.py` (needs `pip install -e .[server]`).
2. Open `clients/unity/` in Unity Hub with 6000.3.3f1 (sign in / activate a license). Let it import
   and confirm the Console has no compile errors. Unity will create `Library/`, `ProjectSettings/*`
   and a scene prompt; open or create any empty scene and press Play. The client builds its own
   world-space card at runtime (`ContextLensClient.Boot`), so no scene setup is needed.
3. The manifest adds the Input System (1.17.0) and XR Interaction Toolkit (3.3.0); both versions are
   in the 6000.3.3f1 install's package cache, so they resolve offline. Installing the Input System
   prompts to enable the new input backend and restart: accept, then set Project Settings > Player >
   Active Input Handling to "Both" (the card's buttons use the legacy event system). Neither package
   has been resolved in the Editor yet; note any Package Manager error.
4. Expected on Play: top line "REC: server says recording" (red), shown only once the server has said
   so. Before that it reads "Not recording (connecting)"; if the server is not running, "Not recording
   (not connected ...)".
5. Click "Test audio": a transcript line appears. Click "Press" three times: cards come back as
   answered, partial, abstained (see the fake server section of the contract). Pressing before any
   transcript shows "Nothing heard yet" and sends nothing.
6. "Pause": the line changes to "Not recording (paused)" immediately, on the press, before any server
   reply. "Test audio" then sends nothing and adds no transcript. "Resume": the line stays "paused"
   until the server answers `recording`, then shows REC. A `recording` status that arrives after Pause
   (or after Delete) without a Resume press is ignored. Same rules as `clients/web/src/session.ts`.
7. "Delete": the line changes to "Not recording (deleting)" at once; when the server answers, transcript
   and card clear, the indicator shows deleted, and the connection closes.
8. "Webcam": a preview appears and a red "WEBCAM ON (local only, not sent)" line shows; click again to
   turn it off. Nothing from the camera is sent over the socket. (Needs a camera and OS permission.)
9. Controller / XR Interaction Simulator. Neither has been run; this is the setup, and what to look for.
   - Package Manager > XR Interaction Toolkit > Samples > import "XR Interaction Simulator" and
     "Starter Assets". (The toolkit's older "XR Device Simulator" sample is the previous generation;
     use the Interaction Simulator.) Drag `XR Interaction Simulator.prefab` into the scene you play.
   - Assets/Scripts/XRInput/ContextLensXRInput.cs listens on an Input System action bound to
     `<XRController>{LeftHand}/{PrimaryButton}` and `<XRController>{RightHand}/{PrimaryButton}`. The
     simulator creates `XRSimulatedController` devices, which derive from the Input System's
     `XRController` and expose `primaryButton` (read from the toolkit 3.3.0 source). A headset
     controller whose layout carries the PrimaryButton usage (A/X on Touch via OpenXR) matches the same
     binding. A real headset also needs XR Plug-in Management with an OpenXR loader enabled; not set up.
   - In Play mode: press `[` or `]` to manipulate a hand, then cycle the quick action with the backtick
     key until the simulator's on-screen label shows Primary Button and press Space. Expected: the same
     as clicking "Press" (after a transcript exists; otherwise "Nothing heard yet"). If the label or keys
     differ in 3.3.0, follow the simulator's on-screen help and record what actually worked.
   - If the Input System package is missing, `ContextLens.XRInput.asmdef` is not compiled (its
     `CONTEXTLENS_INPUT_SYSTEM` define comes from the package version), and the rest still builds; the
     on-card buttons work, the controller press does not.
10. Untrusted text: card text is shown by a `Text` component with rich text off. Check by
    temporarily making a server card contain `<b>x</b>`: it must show literally.

## Honest limits

- Nothing here has been run in Unity. The controller / simulator press, the webcam, the socket and the
  indicator rules are untested; only the C# compile and the schema check have run.
- The indicator logic mirrors the web client but is not unit-tested (no C# test runner here).
- Docs ambiguity found while doing this, not changed: `docs/client-contract.md` says `pause` has "(no
  reply)", yet its status list has `paused` and `tools/fake_server.py` sends one. This client does not
  depend on that: it shows paused on the press, and accepts either behaviour.

- No microphone capture: "Test audio" sends a chunk of silence as a stand-in (the fake server
  ignores audio content). The indicator reflects only what the server last reported, not what this
  client is capturing, because it captures nothing.
- `System.Net.WebSockets.ClientWebSocket` works in the Editor and Windows/Android standalone
  builds, not WebGL.
- JsonUtility cannot represent null, so optional strings (`label`, `suggestion`, `question`,
  `detail`) are empty strings in C#. An empty `question` is sent as `""`, not `null`.
