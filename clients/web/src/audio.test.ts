import { startMic } from './audio';

describe('startMic', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('stops the microphone tracks if the audio context cannot be created', async () => {
    const stop = vi.fn();
    const track = { stop, addEventListener: vi.fn() };
    vi.stubGlobal('navigator', {
      mediaDevices: { getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [track] }) },
    });
    vi.stubGlobal(
      'AudioContext',
      class {
        constructor() {
          throw new Error('no audio context');
        }
      },
    );
    await expect(startMic(() => {}, () => {})).rejects.toThrow('no audio context');
    expect(stop).toHaveBeenCalledTimes(1);
  });
});
