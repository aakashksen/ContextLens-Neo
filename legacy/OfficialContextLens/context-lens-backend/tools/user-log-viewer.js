'use strict';

var childProcess = require('child_process');
var path = require('path');

function safeJsonParse(line) {
  try {
    return { ok: true, value: JSON.parse(line) };
  } catch (err) {
    return { ok: false, error: err };
  }
}

function main() {
  var indexPath = path.join(__dirname, '..', 'index.js');
  var env = Object.assign({}, process.env, {
    AUDIO_TEST: '0',
    BRAIN_SELFTEST: '0'
  });

  var child = childProcess.spawn(process.execPath, [indexPath], {
    stdio: ['inherit', 'pipe', 'pipe'],
    env: env
  });

  var buffer = '';
  var seen = new Set();
  var seenTips = new Set();
  var lastTranscript = null;
  var lastTranscriptAt = 0;
  var lastTipText = null;
  var lastTipAt = 0;
  var transcriptDedupeMs = Number(process.env.USER_LOG_DEDUPE_MS || 2000);
  var tipDedupeMs = Number(process.env.USER_TIP_DEDUPE_MS || 3000);

  child.stdout.on('data', function (chunk) {
    buffer += chunk.toString();
    var lines = buffer.split(/\r?\n/);
    buffer = lines.pop();

    lines.forEach(function (line) {
      var trimmed = line.trim();
      if (!trimmed) {
        return;
      }
      var parsed = safeJsonParse(trimmed);
      if (!parsed.ok) {
        return;
      }
      var frame = parsed.value;
      if (!frame) {
        return;
      }
      if (Array.isArray(frame.transcript_lines)) {
        frame.transcript_lines.forEach(function (item) {
          if (!item || typeof item.line_id !== 'string') {
            return;
          }
          if (seen.has(item.line_id)) {
            return;
          }
          var speaker = typeof item.speaker === 'string' ? item.speaker : 'speaker0';
          var text = typeof item.text === 'string' ? item.text : '';
          var now = Date.now();
          if (text) {
            var currentSig = speaker + '|' + text;
            if (lastTranscript && currentSig === lastTranscript && (now - lastTranscriptAt) < transcriptDedupeMs) {
              return;
            }
            lastTranscript = currentSig;
            lastTranscriptAt = now;
          }
          seen.add(item.line_id);
          if (text) {
            process.stdout.write(speaker + ': ' + text + '\n');
          }
        });
      }

      if (Array.isArray(frame.active_tips)) {
        frame.active_tips.forEach(function (tip) {
          if (!tip || typeof tip.tip_id !== 'string') {
            return;
          }
          if (seenTips.has(tip.tip_id)) {
            return;
          }
          seenTips.add(tip.tip_id);
          var tipText = typeof tip.text === 'string' ? tip.text : '';
          var nowTip = Date.now();
          if (tipText && lastTipText === tipText && (nowTip - lastTipAt) < tipDedupeMs) {
            return;
          }
          if (tipText) {
            lastTipText = tipText;
            lastTipAt = nowTip;
          }
          if (tipText) {
            process.stdout.write('TIP: ' + tipText + '\n');
          }
        });
      }
    });
  });

  if (process.env.USER_LOG_VIEWER_DEBUG === '1') {
    child.stderr.on('data', function (chunk) {
      process.stderr.write(chunk);
    });
  }

  child.on('close', function (code) {
    process.exitCode = code || 0;
  });
}

main();
