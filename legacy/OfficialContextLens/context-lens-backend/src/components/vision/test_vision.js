'use strict';

var fs = require('fs');
var path = require('path');
var VisionService = require('./vision.service');
var logger = require('../../core/logger');

function loadDotEnv() {
  var envPath = path.join(__dirname, '..', '..', '..', '.env');
  if (!fs.existsSync(envPath)) {
    return;
  }
  var raw = fs.readFileSync(envPath, 'utf8');
  raw.split(/\r?\n/).forEach(function (line) {
    var trimmed = line.trim();
    if (!trimmed || trimmed[0] === '#') return;
    var idx = trimmed.indexOf('=');
    if (idx === -1) return;
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

function run() {
  var mode = process.env.VISION_MODE || 'mock';
  var intervalMs = Number(process.env.VISION_INTERVAL_MS || 3000);
  var service = new VisionService({
    mode: mode,
    intervalMs: intervalMs,
    geminiApiKey: process.env.GEMINI_API_KEY,
    geminiModel: process.env.GEMINI_MODEL || 'gemini-1.5-flash',
    captureTimeoutMs: Number(process.env.VISION_CAPTURE_TIMEOUT_MS || 5000)
  });

  service.on('snapshot', function (snap) {
    logger.info('vision snapshot', snap);
  });

  service.start().catch(function (err) {
    logger.error('Vision test failed', { error: err.message });
    process.exit(1);
  });

  var durationMs = Number(process.env.VISION_TEST_DURATION_MS || process.argv[2] || 60000);
  if (Number.isFinite(durationMs) && durationMs > 0) {
    setTimeout(function () {
      service.stop().then(function () {
        process.exit(0);
      });
    }, durationMs);
  } else {
    logger.info('Vision test running until Ctrl+C');
    process.on('SIGINT', function () {
      service.stop().then(function () {
        process.exit(0);
      });
    });
  }
}

run();
