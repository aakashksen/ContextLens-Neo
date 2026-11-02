'use strict';

var EventEmitter = require('events').EventEmitter;
var GeminiVisionClient = require('./gemini.vision').GeminiVisionClient;
var PlaywrightFrameSource = require('./frame_source_playwright').PlaywrightFrameSource;

function VisionService(config) {
  EventEmitter.call(this);
  this.config = config;
  this.latest = null;
  this.running = false;
  this.timer = null;
  this.frameSource = null;
  this.gemini = null;
  this.mockIndex = 0;
  this.backoffUntil = 0;
}

VisionService.prototype = Object.create(EventEmitter.prototype);
VisionService.prototype.constructor = VisionService;

VisionService.prototype.start = async function () {
  if (this.running) return;
  this.running = true;

  var interval = Math.max(1000, this.config.intervalMs);
  if (interval !== this.config.intervalMs) {
    console.warn('VISION_INTERVAL_MS capped to 1000ms minimum');
  }

  if (this.config.mode === 'gemini' && this.config.geminiApiKey) {
    this.gemini = new GeminiVisionClient(this.config.geminiApiKey, this.config.geminiModel || 'gemini-1.5-flash');
    if (!this.gemini.isReady()) {
      console.warn('Gemini not ready, switching to mock mode');
      this.config.mode = 'mock';
    }
  } else {
    this.config.mode = 'mock';
  }

  if (this.config.mode === 'gemini') {
    this.frameSource = new PlaywrightFrameSource(this.config.captureTimeoutMs || 5000);
    try {
      await this.frameSource.start();
    } catch (err) {
      console.warn('Frame source failed to start, switching to mock', err);
      this.config.mode = 'mock';
    }
  }

  var self = this;
  var loop = async function () {
    if (!self.running) return;
    if (Date.now() < self.backoffUntil) {
      self.timer = setTimeout(loop, interval);
      return;
    }
    try {
      var snapshot = self.config.mode === 'mock'
        ? self.mockSnapshot()
        : await self.captureSnapshot();
      if (snapshot) {
        self.latest = snapshot;
        self.emit('snapshot', snapshot);
      }
    } catch (error) {
      console.warn('VisionService tick failed', error);
      var backoffMs = extractBackoffMs(error);
      if (backoffMs) {
        self.backoffUntil = Date.now() + backoffMs;
      }
      var degraded = self.latest || self.fallbackSnapshot();
      self.latest = {
        ts_ms: Date.now(),
        environment: degraded.environment,
        people: degraded.people,
        social_cues: degraded.social_cues,
        reliability: {
          overall_confidence: degraded.reliability.overall_confidence || 0.2,
          limitations: dedupeNotes((degraded.reliability.limitations || []).concat(['capture_error']))
        },
        notes: dedupeNotes((degraded.notes || []).concat(['vision_degraded']))
      };
      self.emit('snapshot', self.latest);
    }
    self.timer = setTimeout(loop, interval);
  };

  this.timer = setTimeout(loop, 0);
};

VisionService.prototype.stop = async function () {
  this.running = false;
  if (this.timer) {
    clearTimeout(this.timer);
    this.timer = null;
  }
  if (this.frameSource) {
    await this.frameSource.stop();
    this.frameSource = null;
  }
};

VisionService.prototype.getLatest = function () {
  return this.latest;
};

VisionService.prototype.captureSnapshot = async function () {
  if (!this.frameSource || !this.gemini) return null;
  var buffer = await this.frameSource.captureJpegBuffer();
  var snapshot = await this.gemini.sendImage(buffer);
  snapshot.frame_id = snapshot.frame_id || ('frame-' + Date.now());
  return snapshot;
};

VisionService.prototype.mockSnapshot = function () {
  var envs = ['study room', 'classroom', 'office'];
  var objects = ['laptop', 'desk', 'whiteboard'];
  var envLabel = envs[this.mockIndex % envs.length];
  var object = objects[this.mockIndex % objects.length];
  var envConfidence = seededConfidence(this.mockIndex);
  this.mockIndex += 1;
  return {
    ts_ms: Date.now(),
    environment: {
      label: envLabel,
      confidence: envConfidence,
      objects: [object],
      lighting: {
        level: 'normal',
        source: 'indoor',
        confidence: 0.6
      },
      noise_or_busyness: {
        level: 'moderate',
        confidence: 0.5
      }
    },
    people: {
      count_estimate: this.mockIndex % 2,
      count_confidence: 0.4,
      proximity: 'medium',
      orientation_summary: 'mixed',
      confidence: 0.4
    },
    social_cues: {
      facial_expression_summary: {
        label: 'unknown',
        confidence: 0.2,
        notes: ['mock_mode']
      },
      body_posture_summary: {
        label: 'unknown',
        confidence: 0.2
      },
      gaze_summary: {
        label: 'unknown',
        confidence: 0.2
      },
      interaction_context: {
        label: 'unknown',
        confidence: 0.2
      }
    },
    reliability: {
      overall_confidence: 0.4,
      limitations: ['mock_mode']
    },
    notes: ['mock_mode']
  };
};

VisionService.prototype.fallbackSnapshot = function () {
  return {
    ts_ms: Date.now(),
    environment: {
      label: 'unknown',
      confidence: 0.2,
      objects: [],
      lighting: {
        level: 'normal',
        confidence: 0.4
      }
    },
    people: {
      count_estimate: 0,
      count_confidence: 0.2,
      proximity: 'medium',
      confidence: 0.2
    },
    social_cues: {
      facial_expression_summary: {
        label: 'unknown',
        confidence: 0.2
      },
      body_posture_summary: {
        label: 'unknown',
        confidence: 0.2
      },
      gaze_summary: {
        label: 'unknown',
        confidence: 0.2
      },
      interaction_context: {
        label: 'unknown',
        confidence: 0.2
      }
    },
    reliability: {
      overall_confidence: 0.2,
      limitations: ['no_snapshot']
    },
    notes: ['no_snapshot']
  };
};

function seededConfidence(seed) {
  var x = Math.sin(seed + 1) * 10000;
  var value = x - Math.floor(x);
  return Number((0.5 + value * 0.4).toFixed(2));
}

function dedupeNotes(notes) {
  var set = new Set(notes);
  return Array.from(set).slice(0, 3);
}

function extractBackoffMs(error) {
  var message = error instanceof Error ? error.message : String(error);
  if (!message) return null;
  var retryMatch = message.match(/retryDelay\":\"(\d+)s\"/);
  if (retryMatch) return Number(retryMatch[1]) * 1000;
  if (message.indexOf('429') >= 0 || message.indexOf('Too Many Requests') >= 0) {
    return 15000;
  }
  return null;
}

module.exports = VisionService;
