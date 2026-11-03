# Regenerates tests/fixtures/speech/two_voices.wav and two_voices.json: four spoken turns that alternate between two
# Windows built-in synthetic voices (David, Zira) with one second of silence between them; 16 kHz mono 16-bit.
# The JSON lists each turn's voice and its start and end in milliseconds.
Add-Type -AssemblyName System.Speech
$outDir = Join-Path $PSScriptRoot '..\tests\fixtures\speech'
$tmp = Join-Path ([IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path $tmp | Out-Null
$turns = @(
  @('Microsoft David Desktop', 'The quarterly budget review is scheduled for Thursday afternoon.'),
  @('Microsoft Zira Desktop',  'I will send the updated spreadsheet to everyone before then.'),
  @('Microsoft David Desktop', 'Good, and please include the travel costs from last month.'),
  @('Microsoft Zira Desktop',  'Of course, I will add a separate tab for those.')
)
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$pcm = New-Object System.IO.MemoryStream
$meta = @()
$silence = New-Object byte[] 32000
for ($i = 0; $i -lt $turns.Count; $i++) {
  $f = Join-Path $tmp "t$i.wav"
  $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
  $s.SelectVoice($turns[$i][0])
  $s.SetOutputToWaveFile($f, $fmt)
  $s.Speak($turns[$i][1])
  $s.Dispose()
  $b = [IO.File]::ReadAllBytes($f)
  $start = [int]($pcm.Length / 32)
  $pcm.Write($b, 44, $b.Length - 44)
  $meta += [ordered]@{ voice = $turns[$i][0]; start_ms = $start; end_ms = [int]($pcm.Length / 32) }
  $pcm.Write($silence, 0, $silence.Length)
  Remove-Item -LiteralPath $f
}
Remove-Item -LiteralPath $tmp
$data = $pcm.ToArray()
$w = New-Object System.IO.BinaryWriter([IO.File]::Create((Join-Path $outDir 'two_voices.wav')))
$w.Write([Text.Encoding]::ASCII.GetBytes('RIFF')); $w.Write([int](36 + $data.Length))
$w.Write([Text.Encoding]::ASCII.GetBytes('WAVEfmt ')); $w.Write([int]16); $w.Write([int16]1); $w.Write([int16]1)
$w.Write([int]16000); $w.Write([int]32000); $w.Write([int16]2); $w.Write([int16]16)
$w.Write([Text.Encoding]::ASCII.GetBytes('data')); $w.Write([int]$data.Length); $w.Write($data)
$w.Close()
ConvertTo-Json -InputObject $meta | Set-Content -Encoding ascii (Join-Path $outDir 'two_voices.json')
