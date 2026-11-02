'use strict';

var VisionService = require('./vision.service');

function summarizeSnapshot(snapshot) {
  if (!snapshot) return 'vision unavailable';
  var env = snapshot.environment || {};
  var people = snapshot.people || {};
  var cues = snapshot.social_cues || {};
  var parts = [];
  if (env.label) {
    parts.push(env.label);
  }
  if (Array.isArray(env.objects) && env.objects.length) {
    parts.push('objects: ' + env.objects.slice(0, 4).join(', '));
  }
  if (Number.isFinite(people.count_estimate)) {
    parts.push('people: ' + people.count_estimate);
  }
  if (cues.interaction_context && cues.interaction_context.label) {
    parts.push('context: ' + cues.interaction_context.label);
  }
  return parts.length ? parts.join(' | ') : 'vision snapshot';
}

function startVision(options) {
  var state = options.state;
  var logger = options.logger;
  var mode = process.env.VISION_MODE || 'mock';
  var intervalMs = Number(process.env.VISION_INTERVAL_MS || 3000);
  var captureTimeoutMs = Number(process.env.VISION_CAPTURE_TIMEOUT_MS || 5000);
  var testEnabled = process.env.VISION_TEST === '1';
  var testIntervalMs = Number(process.env.VISION_TEST_INTERVAL_MS || intervalMs);
  var testTimer = null;

  var service = new VisionService({
    mode: mode,
    intervalMs: intervalMs,
    geminiApiKey: process.env.GEMINI_API_KEY,
    geminiModel: process.env.GEMINI_MODEL || 'gemini-1.5-flash',
    captureTimeoutMs: captureTimeoutMs
  });

  service.on('snapshot', function (snapshot) {
    var event = {
      event_id: 'vision_' + snapshot.ts_ms,
      event_type: 'vision',
      ts_ms: snapshot.ts_ms,
      payload: {
        scene_summary: summarizeSnapshot(snapshot)
      }
    };
    try {
      state.ingest(event);
    } catch (err) {
      logger.error('Vision ingest failed', { error: err.message });
    }
  });

  service.start().catch(function (err) {
    logger.error('Vision service failed to start', { error: err.message });
  });

  if (testEnabled) {
    logger.info('Vision test mode enabled', { intervalMs: testIntervalMs });
    testTimer = setInterval(function () {
      logger.info('Vision test heartbeat');
    }, testIntervalMs);
  }

  return function stop() {
    service.stop().catch(function (err) {
      logger.error('Vision service stop failed', { error: err.message });
    });
    if (testTimer) {
      clearInterval(testTimer);
    }
  };
}

module.exports = {
  startVision: startVision
};
