'use strict';

var https = require('https');
var prompt = require('./prompt');

function isObject(val) {
  return val !== null && typeof val === 'object' && !Array.isArray(val);
}

function stripCodeFences(text) {
  var trimmed = String(text || '').trim();
  if (trimmed.indexOf('```') !== 0) {
    return trimmed;
  }
  return trimmed.replace(/^```[a-zA-Z0-9_-]*\s*/m, '').replace(/```$/m, '').trim();
}

function validateBrainOutput(brainJson) {
  if (!isObject(brainJson) || typeof brainJson.action !== 'string') {
    return { ok: false, error: 'Brain output must include action' };
  }

  if (brainJson.action === 'SKIP') {
    return { ok: true, value: brainJson };
  }

  if (brainJson.action !== 'UPDATE') {
    return { ok: false, error: 'Brain output action must be SKIP or UPDATE' };
  }

  if (!Array.isArray(brainJson.transcript_add)) {
    return { ok: false, error: 'transcript_add must be an array' };
  }
  if (!Array.isArray(brainJson.tips_add)) {
    return { ok: false, error: 'tips_add must be an array' };
  }
  if (!Array.isArray(brainJson.tips_remove)) {
    return { ok: false, error: 'tips_remove must be an array' };
  }
  if (!Array.isArray(brainJson.suggested_responses)) {
    return { ok: false, error: 'suggested_responses must be an array' };
  }

  for (var i = 0; i < brainJson.transcript_add.length; i++) {
    var line = brainJson.transcript_add[i];
    if (!isObject(line) || typeof line.line_id !== 'string' || typeof line.speaker !== 'string' || typeof line.text !== 'string') {
      return { ok: false, error: 'Invalid transcript_add item' };
    }
  }

  for (var j = 0; j < brainJson.tips_add.length; j++) {
    var tip = brainJson.tips_add[j];
    if (!isObject(tip) || typeof tip.tip_id !== 'string' || typeof tip.text !== 'string') {
      return { ok: false, error: 'Invalid tips_add item' };
    }
  }

  for (var k = 0; k < brainJson.tips_remove.length; k++) {
    if (typeof brainJson.tips_remove[k] !== 'string') {
      return { ok: false, error: 'Invalid tips_remove item' };
    }
  }

  for (var m = 0; m < brainJson.suggested_responses.length; m++) {
    if (typeof brainJson.suggested_responses[m] !== 'string') {
      return { ok: false, error: 'Invalid suggested_responses item' };
    }
  }

  return { ok: true, value: brainJson };
}

function buildTranscriptAdd(snapshot, processedIds, counterRef) {
  if (!snapshot || !Array.isArray(snapshot.recent_sensor_events)) {
    return [];
  }

  var additions = [];
  snapshot.recent_sensor_events.forEach(function (evt) {
    if (!evt || typeof evt.event_id !== 'string') {
      return;
    }
    if (processedIds.has(evt.event_id)) {
      return;
    }
    var isTranscript = evt.event_type === 'transcript_delta' || evt.type === 'TRANSCRIPT_DELTA';
    if (!isTranscript || !evt.payload) {
      return;
    }
    if (evt.payload.is_final === false) {
      return;
    }
    processedIds.add(evt.event_id);
    counterRef.value += 1;
    additions.push({
      line_id: 't' + counterRef.value,
      speaker: typeof evt.payload.speaker === 'string' ? evt.payload.speaker : 'speaker0',
      text: typeof evt.payload.text === 'string' ? evt.payload.text : ''
    });
  });

  return additions.filter(function (line) { return line.text; });
}

function buildTipOnVision(snapshot, processedVisionIds) {
  if (!snapshot || !Array.isArray(snapshot.recent_sensor_events)) {
    return [];
  }
  var newVision = snapshot.recent_sensor_events.find(function (evt) {
    return evt && evt.event_type === 'vision' && typeof evt.event_id === 'string' && !processedVisionIds.has(evt.event_id);
  });
  if (!newVision) {
    return [];
  }
  processedVisionIds.add(newVision.event_id);
  return [{
    tip_id: 'tip_vision_' + newVision.event_id,
    text: 'Test tip: vision frame received.'
  }];
}

function geminiRequest(apiKey, model, payload, timeoutMs, cb) {
  var data = JSON.stringify(payload);
  var req = https.request({
    hostname: 'generativelanguage.googleapis.com',
    path: '/v1beta/models/' + encodeURIComponent(model) + ':generateContent?key=' + encodeURIComponent(apiKey),
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(data)
    },
    timeout: timeoutMs
  }, function (res) {
    var body = '';
    res.setEncoding('utf8');
    res.on('data', function (chunk) { body += chunk; });
    res.on('end', function () {
      cb(null, { status: res.statusCode, body: body });
    });
  });

  req.on('error', function (err) {
    cb(err);
  });

  req.on('timeout', function () {
    req.destroy(new Error('Gemini request timeout'));
  });

  req.write(data);
  req.end();
}

function runSelfTest(logger) {
  var pass = 0;
  var fail = 0;

  function check(name, obj, shouldPass) {
    var result = validateBrainOutput(obj);
    var ok = result.ok === shouldPass;
    if (ok) {
      pass += 1;
      logger.info('Brain selftest pass', { name: name });
    } else {
      fail += 1;
      logger.error('Brain selftest fail', { name: name, error: result.error || 'unexpected result' });
    }
  }

  check('skip', { action: 'SKIP' }, true);
  check('update', { action: 'UPDATE', transcript_add: [], tips_add: [], tips_remove: [], suggested_responses: [] }, true);
  check('bad_action', { action: 'NOPE' }, false);
  check('bad_update', { action: 'UPDATE', transcript_add: [{}], tips_add: [], tips_remove: [], suggested_responses: [] }, false);

  return { pass: pass, fail: fail };
}

function startBrain(options) {
  var state = options.state;
  var logger = options.logger;
  var timings = options.timings || {};
  var intervalMs = Number(timings.brain_tick_ms || 1500);
  var apiKey = process.env.GEMINI_API_KEY;
  var model = process.env.GEMINI_MODEL || 'gemini-3-flash-preview';
  var enabled = process.env.SIM_BRAIN === '1';
  var testEnabled = process.env.BRAIN_TEST === '1';
  var selfTest = process.env.BRAIN_SELFTEST === '1';
  var testIntervalMs = Number(process.env.BRAIN_TEST_INTERVAL_MS || intervalMs);
  var testTimer = null;
  var timer = null;
  var transcriptCounter = 0;
  var processedEventIds = new Set();
  var processedVisionIds = new Set();
  var transcriptCounterRef = { value: 0 };

  if (selfTest) {
    logger.info('Brain selftest enabled');
    var result = runSelfTest(logger);
    logger.info('Brain selftest complete', result);
    return function stop() {};
  }

  if (enabled) {
    logger.info('Brain sim enabled', { intervalMs: intervalMs });
  } else {
    logger.info('Brain sim disabled');
  }

  if (testEnabled) {
    logger.info('Brain test mode enabled', { intervalMs: testIntervalMs });
    testTimer = setInterval(function () {
      logger.info('Brain test heartbeat');
    }, testIntervalMs);
  }

  if (enabled) {
    timer = setInterval(function () {
      var snapshot = state.snapshotForBrain();
      var transcript_add = buildTranscriptAdd(snapshot, processedEventIds, transcriptCounterRef);

      if (transcript_add.length === 0) {
        try {
          state.applyBrainOutput({ action: 'SKIP' });
        } catch (err) {
          logger.error('Brain SKIP failed', { error: err.message });
        }
        return;
      }

      var update = {
        action: 'UPDATE',
        transcript_add: transcript_add,
        tips_add: buildTipOnVision(snapshot, processedVisionIds),
        tips_remove: [],
        suggested_responses: []
      };

      try {
        state.applyBrainOutput(update);
      } catch (err) {
        logger.error('Brain UPDATE failed', { error: err.message });
      }
    }, intervalMs);
  } else if (apiKey) {
    timer = setInterval(function () {
      var snapshot = state.snapshotForBrain();
      var pendingTranscript = buildTranscriptAdd(snapshot, processedEventIds, transcriptCounterRef);
      var inputText = prompt + '\n\nBrain Input Snapshot:\n' + JSON.stringify(snapshot);
      var payload = {
        contents: [
          { role: 'user', parts: [{ text: inputText }] }
        ],
        generationConfig: {
          temperature: 0.2
        }
      };

      geminiRequest(apiKey, model, payload, 10000, function (err, res) {
        if (err) {
          logger.error('Gemini request failed', { error: err.message });
          return;
        }

        if (!res || res.status < 200 || res.status >= 300) {
          logger.error('Gemini non-2xx response', { status: res ? res.status : 0, body: res ? res.body : '' });
          return;
        }

        var parsed;
        try {
          parsed = JSON.parse(res.body);
        } catch (parseErr) {
          logger.error('Gemini response JSON parse failed', { error: parseErr.message });
          return;
        }

        var candidates = parsed.candidates || [];
        var text = '';
        if (candidates[0] && candidates[0].content && Array.isArray(candidates[0].content.parts)) {
          text = candidates[0].content.parts.map(function (p) { return p.text || ''; }).join('');
        }

        if (!text) {
          logger.error('Gemini response missing text');
          return;
        }

        var cleaned = stripCodeFences(text);
        var brainJson;
        try {
          brainJson = JSON.parse(cleaned);
        } catch (err2) {
          logger.error('Gemini output JSON parse failed', { error: err2.message, text: cleaned });
          return;
        }

        var validated = validateBrainOutput(brainJson);
        if (!validated.ok) {
          logger.error('Gemini output validation failed', { error: validated.error });
          if (pendingTranscript.length > 0) {
            try {
              state.applyBrainOutput({
                action: 'UPDATE',
                transcript_add: pendingTranscript,
                tips_add: [],
                tips_remove: [],
                suggested_responses: []
              });
            } catch (applyErr) {
              logger.error('Apply fallback brain output failed', { error: applyErr.message });
            }
          }
          return;
        }

        if (validated.value.action === 'SKIP' && pendingTranscript.length > 0) {
          validated.value = {
            action: 'UPDATE',
            transcript_add: pendingTranscript,
            tips_add: buildTipOnVision(snapshot, processedVisionIds),
            tips_remove: [],
            suggested_responses: []
          };
        }

        if (validated.value.action === 'UPDATE') {
          if (!Array.isArray(validated.value.transcript_add) || validated.value.transcript_add.length === 0) {
            if (pendingTranscript.length > 0) {
              validated.value.transcript_add = pendingTranscript;
            }
          }
          if (!Array.isArray(validated.value.tips_add) || validated.value.tips_add.length === 0) {
            validated.value.tips_add = buildTipOnVision(snapshot, processedVisionIds);
          }
          validated.value.tips_remove = [];
          if (!Array.isArray(validated.value.suggested_responses)) {
            validated.value.suggested_responses = [];
          }
        }

        try {
          state.applyBrainOutput(validated.value);
        } catch (applyErr) {
          logger.error('Apply brain output failed', { error: applyErr.message });
        }
      });
    }, intervalMs);
  } else {
    logger.warn('Gemini disabled (missing GEMINI_API_KEY)');
  }

  return function stop() {
    if (timer) {
      clearInterval(timer);
    }
    if (testTimer) {
      clearInterval(testTimer);
    }
  };
}

module.exports = {
  startBrain: startBrain,
  validateBrainOutput: validateBrainOutput
};
