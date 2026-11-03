using UnityEngine;
using UnityEngine.InputSystem;

namespace ContextLens
{
    // Calls ContextLensClient.Press when the primary button (A/X) of either hand goes down.
    // The binding is by XR usage, so it matches an OpenXR controller and the XR Interaction Simulator's
    // simulated controller alike (XRSimulatedController derives from the Input System's XRController).
    // This assembly compiles only while the Input System package is installed (see the asmdef), so a
    // missing package leaves the rest of the client building and the on-card buttons working.
    public class ContextLensXRInput : MonoBehaviour
    {
        InputAction primary;
        ContextLensClient client;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Boot()
        {
            if (FindFirstObjectByType<ContextLensXRInput>() != null) return;
            new GameObject("ContextLensXRInput").AddComponent<ContextLensXRInput>();
        }

        void OnEnable()
        {
            primary = new InputAction("ContextLensPrimary", InputActionType.Button);
            primary.AddBinding("<XRController>{LeftHand}/{PrimaryButton}");
            primary.AddBinding("<XRController>{RightHand}/{PrimaryButton}");
            primary.started += OnPrimary;
            primary.Enable();
        }

        void OnDisable()
        {
            primary.started -= OnPrimary;
            primary.Dispose();
        }

        void OnPrimary(InputAction.CallbackContext _)
        {
            if (client == null) client = FindFirstObjectByType<ContextLensClient>();
            if (client != null) client.Press();
        }
    }
}
