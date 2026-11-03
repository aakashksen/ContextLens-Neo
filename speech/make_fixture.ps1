# Regenerates tests/fixtures/speech/phrase.wav with the Windows built-in synthetic voice (Microsoft David), 16 kHz mono 16-bit.
Add-Type -AssemblyName System.Speech
$out = Join-Path $PSScriptRoot '..\tests\fixtures\speech\phrase.wav'
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice('Microsoft David Desktop')
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$s.SetOutputToWaveFile($out, $fmt)
$s.Speak('The quarterly budget review is scheduled for Thursday afternoon.')
$s.Dispose()
