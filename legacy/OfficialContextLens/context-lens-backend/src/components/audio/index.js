'use strict';

var childProcess = require('child_process');

function safeJsonParse(line) {
  try {
    return { ok: true, value: JSON.parse(line) };
  } catch (err) {
    return { ok: false, error: err };
  }
}

function normalizeTranscriptPayload(input) {
  if (input && typeof input === 'object') {
    return {
      speaker: typeof input.speaker === 'string' ? input.speaker : 'speaker0',
      text: typeof input.text === 'string' ? input.text : '',
      confidence: typeof input.confidence === 'number' ? input.confidence : 0,
      is_final: typeof input.is_final === 'boolean' ? input.is_final : true
    };
  }

  return {
    speaker: 'speaker0',
    text: String(input || ''),
    confidence: 0,
    is_final: true
  };
}

function buildTranscriptEvent(counter, payload, timestampMs) {
  var ts = Number.isFinite(timestampMs) ? timestampMs : Date.now();
  return {
    event_id: 'transcript_' + counter,
    event_type: 'transcript_delta',
    ts_ms: ts,
    type: 'TRANSCRIPT_DELTA',
    timestamp_ms: ts,
    payload: payload
  };
}

function startStdinIngest(options) {
  var state = options.state;
  var logger = options.logger;
  var counter = 0;
  var enabled = process.env.AUDIO_STDIN === '1';

  if (!enabled) {
    return function stop() {};
  }

  logger.info('Audio stdin ingest enabled');

  if (process.stdin.isTTY) {
    logger.info('Waiting for stdin lines (JSON or plain text)');
  }

  process.stdin.setEncoding('utf8');

  function onData(chunk) {
    var lines = chunk.split(/\r?\n/);
    lines.forEach(function (line) {
      var trimmed = line.trim();
      if (!trimmed) {
        return;
      }

      var parsed = safeJsonParse(trimmed);
      var payload = parsed.ok ? normalizeTranscriptPayload(parsed.value) : normalizeTranscriptPayload(trimmed);
      var timestampMs = parsed.ok && Number.isFinite(parsed.value.timestamp_ms) ? parsed.value.timestamp_ms : Date.now();

      counter += 1;
      var event = buildTranscriptEvent(counter, payload, timestampMs);

      try {
        state.ingest(event);
        logger.info('Audio stdin transcript', { text: payload.text, speaker: payload.speaker, is_final: payload.is_final });
      } catch (err) {
        logger.error('Audio stdin ingest failed', { error: err.message });
      }
    });
  }

  process.stdin.on('data', onData);
  process.stdin.resume();

  return function stop() {
    process.stdin.removeListener('data', onData);
  };
}

function startSim(options) {
  var state = options.state;
  var logger = options.logger;
  var enabled = process.env.SIM_AUDIO === '1';
  var intervalMs = Number(process.env.SIM_AUDIO_INTERVAL_MS || 2000);
  var timer = null;
  var counter = 0;

  if (!enabled) {
    return function stop() {};
  }

  logger.info('Audio sim enabled', { intervalMs: intervalMs });

  timer = setInterval(function () {
    counter += 1;
    var payload = {
      speaker: counter % 2 === 0 ? 'speaker1' : 'speaker0',
      text: 'simulated utterance #' + counter,
      confidence: 0.9,
      is_final: true
    };
    var event = buildTranscriptEvent(counter, payload, Date.now());

    try {
      state.ingest(event);
      logger.info('Audio sim transcript', { text: payload.text, speaker: payload.speaker, is_final: payload.is_final });
    } catch (err) {
      logger.error('Audio ingest failed', { error: err.message });
    }
  }, intervalMs);

  return function stop() {
    if (timer) {
      clearInterval(timer);
    }
  };
}

function deriveSpeaker(words) {
  if (!Array.isArray(words) || words.length === 0) {
    return 'speaker0';
  }
  var counts = {};
  words.forEach(function (w) {
    if (typeof w.speaker === 'number') {
      var key = String(w.speaker);
      counts[key] = (counts[key] || 0) + 1;
    }
  });
  var best = null;
  Object.keys(counts).forEach(function (key) {
    if (!best || counts[key] > counts[best]) {
      best = key;
    }
  });
  return best ? 'speaker' + best : 'speaker0';
}

function extractUtterance(msg) {
  if (!msg || typeof msg !== 'object') {
    return null;
  }

  if (msg.utterance && typeof msg.utterance === 'object') {
    return {
      transcript: msg.utterance.transcript || msg.utterance.text || '',
      words: Array.isArray(msg.utterance.words) ? msg.utterance.words : [],
      speaker: msg.utterance.speaker
    };
  }

  if (typeof msg.transcript === 'string' && msg.type && String(msg.type).toLowerCase().indexOf('utterance') !== -1) {
    return {
      transcript: msg.transcript,
      words: Array.isArray(msg.words) ? msg.words : [],
      speaker: msg.speaker
    };
  }

  return null;
}

function startDeepgram(options) {
  var state = options.state;
  var logger = options.logger;
  var apiKey = process.env.DEEPGRAM_API_KEY;
  var counter = 0;
  var ws = null;
  var sox = null;
  var lastTranscript = null;
  var lastTranscriptAt = 0;
  var dedupeWindowMs = 2000;

  if (!apiKey) {
    return function stop() {};
  }

  var WebSocketCtor;
  try {
    WebSocketCtor = require('ws');
  } catch (err) {
    logger.error('Deepgram requires ws dependency (npm i ws)');
    return function stop() {};
  }

  var url = process.env.DEEPGRAM_URL ||
    'wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=16000&channels=1&punctuate=true&diarize=true&utterances=true&interim_results=true';

  logger.info('Connecting to Deepgram', { url: url });

  ws = new WebSocketCtor(url, {
    headers: {
      Authorization: 'Token ' + apiKey
    }
  });

  ws.on('open', function () {
    logger.info('Deepgram socket open');
    var soxPath = process.env.SOX_PATH || 'sox';
    var device = process.env.SOX_WAVEAUDIO_DEVICE || '1';
    var args;

    if (process.platform === 'win32') {
      args = ['-t', 'waveaudio', device, '-r', '16000', '-c', '1', '-b', '16', '-e', 'signed-integer', '-t', 'raw', '-'];
    } else {
      args = ['-d', '-r', '16000', '-c', '1', '-b', '16', '-e', 'signed-integer', '-t', 'raw', '-'];
    }

    sox = childProcess.spawn(soxPath, args, { stdio: ['ignore', 'pipe', 'pipe'] });

    sox.stdout.on('data', function (chunk) {
      if (ws.readyState === WebSocketCtor.OPEN) {
        ws.send(chunk);
      }
    });

    sox.stderr.on('data', function (data) {
      var msg = String(data).trim();
      if (!msg) {
        return;
      }
      if (process.env.SOX_LOG === '1') {
        logger.warn('sox stderr', { message: msg });
      }
    });

    sox.on('close', function (code) {
      logger.info('sox closed', { code: code });
    });

    sox.on('error', function (err) {
      logger.error('sox failed to start', { error: err.message });
    });
  });

  ws.on('message', function (data) {
    var text = data.toString();
    var parsed = safeJsonParse(text);
    if (!parsed.ok) {
      logger.warn('Deepgram message parse failed', { error: parsed.error.message });
      return;
    }

    var msg = parsed.value;
    var utterance = extractUtterance(msg);
    var transcript = '';
    var speaker = 'speaker0';
    var confidence = 0;
    var isFinal = true;

    if (utterance) {
      transcript = typeof utterance.transcript === 'string' ? utterance.transcript.trim() : '';
      speaker = typeof utterance.speaker === 'number' ? 'speaker' + utterance.speaker : deriveSpeaker(utterance.words);
    } else if (msg && msg.channel && Array.isArray(msg.channel.alternatives) && msg.is_final === true) {
      var alt = msg.channel.alternatives[0] || {};
      transcript = typeof alt.transcript === 'string' ? alt.transcript.trim() : '';
      speaker = deriveSpeaker(alt.words);
      confidence = typeof alt.confidence === 'number' ? alt.confidence : 0;
      isFinal = true;
    } else {
      return;
    }

    if (!transcript) {
      return;
    }

    var nowMs = Date.now();
    var currentSig = speaker + '|' + transcript;
    if (lastTranscript && currentSig === lastTranscript && (nowMs - lastTranscriptAt) < dedupeWindowMs) {
      return;
    }
    lastTranscript = currentSig;
    lastTranscriptAt = nowMs;

    var payload = {
      speaker: speaker,
      text: transcript,
      confidence: confidence,
      is_final: isFinal
    };

    counter += 1;
    var event = buildTranscriptEvent(counter, payload, Date.now());

    try {
      state.ingest(event);
      logger.info('Deepgram transcript', { text: payload.text, speaker: payload.speaker, is_final: payload.is_final });
    } catch (err) {
      logger.error('Deepgram ingest failed', { error: err.message });
    }
  });

  ws.on('close', function () {
    logger.info('Deepgram socket closed');
  });

  ws.on('error', function (err) {
    logger.error('Deepgram socket error', { error: err.message });
  });

  return function stop() {
    if (sox) {
      sox.kill();
    }
    if (ws) {
      ws.close();
    }
  };
}

function startAudio(options) {
  var stopDeepgram = startDeepgram(options);
  var stopStdin = startStdinIngest(options);
  var stopSim = startSim(options);
  var logger = options.logger;
  var testEnabled = process.env.AUDIO_TEST === '1';
  var testIntervalMs = Number(process.env.AUDIO_TEST_INTERVAL_MS || 2000);
  var testTimer = null;

  if (testEnabled) {
    logger.info('Audio test mode enabled', { intervalMs: testIntervalMs });
    testTimer = setInterval(function () {
      logger.info('Audio test heartbeat');
    }, testIntervalMs);
  }

  return function stop() {
    stopDeepgram();
    stopStdin();
    stopSim();
    if (testTimer) {
      clearInterval(testTimer);
    }
  };
}

module.exports = {
  startAudio: startAudio
};
