'use strict';

var fs = require('fs');
var path = require('path');

function loadDotEnv() {
  var envPath = path.join(__dirname, '.env');
  if (!fs.existsSync(envPath)) {
    return;
  }

  var raw = fs.readFileSync(envPath, 'utf8');
  raw.split(/\r?\n/).forEach(function (line) {
    var trimmed = line.trim();
    if (!trimmed || trimmed[0] === '#') {
      return;
    }
    var idx = trimmed.indexOf('=');
    if (idx === -1) {
      return;
    }
    var key = trimmed.slice(0, idx).trim();
    var val = trimmed.slice(idx + 1).trim();
    if (val[0] === '"' && val[val.length - 1] === '"') {
      val = val.slice(1, -1);
    }
    if (!(key in process.env)) {
      process.env[key] = val;
    }
  });
}

loadDotEnv();

var State = require('./src/core/state');
var logger = require('./src/core/logger');
var audio = require('./src/components/audio');
var vision = require('./src/components/vision');
var brain = require('./src/components/brain');

function loadTimings() {
  var timingsPath = path.join(__dirname, 'config', 'timings.json');
  try {
    var raw = fs.readFileSync(timingsPath, 'utf8');
    return JSON.parse(raw);
  } catch (err) {
    logger.error('Failed to load timings.json', { error: err.message });
    return { hud_tick_ms: 500, brain_tick_ms: 1500 };
  }
}

function main() {
  var timings = loadTimings();
  var state = new State({ windowSize: 200 });

  var stopAudio = audio.startAudio({ state: state, logger: logger });
  var stopVision = vision.startVision({ state: state, logger: logger });
  var stopBrain = brain.startBrain({ state: state, logger: logger, timings: timings });

  var hudInterval = Number(timings.hud_tick_ms || 500);
  var hudTimer = setInterval(function () {
    try {
      var frame = state.tickHud(Date.now());
      process.stdout.write(JSON.stringify(frame) + '\n');
    } catch (err) {
      logger.error('HUD tick failed', { error: err.message });
    }
  }, hudInterval);

  function shutdown() {
    logger.info('Shutting down');
    try { stopAudio(); } catch (err) { logger.error('Stop audio failed', { error: err.message }); }
    try { stopVision(); } catch (err) { logger.error('Stop vision failed', { error: err.message }); }
    try { stopBrain(); } catch (err) { logger.error('Stop brain failed', { error: err.message }); }
    clearInterval(hudTimer);
    process.exit(0);
  }

  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}

main();
