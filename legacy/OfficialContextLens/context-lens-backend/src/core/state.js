'use strict';

// TODO: Verify JSON contracts against the Prompt 1 source-of-truth paste.

function isObject(val) {
  return val !== null && typeof val === 'object' && !Array.isArray(val);
}

function clone(obj) {
  return JSON.parse(JSON.stringify(obj));
}

function assertArray(val, name) {
  if (!Array.isArray(val)) {
    throw new Error('Expected array for ' + name);
  }
}

function nowMs() {
  return Date.now();
}

function State(options) {
  var opts = options || {};
  this._windowSize = Number.isFinite(opts.windowSize) ? opts.windowSize : 200;
  this._lineIdSet = new Set();
  this._tipIdSet = new Set();

  this.state_version = 1;
  this.current_frame_id = 0;
  this.last_brain_tick_ms = 0;
  this.committed_transcript_cursor = -1;
  this.transcript_lines = [];
  this.active_tips = [];
  this.suggested_responses = [];
  this.scene_summary = '';
  this.signals = {
    pace_wpm: 0,
    interruptions_count: 0,
    tone_label: ''
  };
  this.pending_sensor_events = [];
}

State.prototype.ingest = function ingest(event) {
  if (!isObject(event)) {
    throw new Error('Sensor event must be an object');
  }
  if (typeof event.event_id !== 'string') {
    throw new Error('Sensor event requires event_id');
  }
  if (typeof event.event_type !== 'string') {
    throw new Error('Sensor event requires event_type');
  }
  if (!Number.isFinite(event.ts_ms)) {
    throw new Error('Sensor event requires ts_ms');
  }
  if (!isObject(event.payload)) {
    throw new Error('Sensor event requires payload');
  }

  this.pending_sensor_events.push(event);

  if (event.event_type === 'vision' && typeof event.payload.scene_summary === 'string') {
    this.scene_summary = event.payload.scene_summary;
  }

  if (event.event_type === 'signals') {
    if (Number.isFinite(event.payload.pace_wpm)) {
      this.signals.pace_wpm = event.payload.pace_wpm;
    }
    if (Number.isFinite(event.payload.interruptions_count)) {
      this.signals.interruptions_count = event.payload.interruptions_count;
    }
    if (typeof event.payload.tone_label === 'string') {
      this.signals.tone_label = event.payload.tone_label;
    }
  }
};

State.prototype.snapshotForBrain = function snapshotForBrain() {
  return {
    frame_id_on_screen: this.current_frame_id,
    committed_transcript_cursor: this.committed_transcript_cursor,
    active_tip_ids: this.active_tips.map(function (t) { return t.tip_id; }),
    transcript_lines: clone(this.transcript_lines),
    scene_summary: this.scene_summary,
    signals: clone(this.signals),
    recent_sensor_events: clone(this.pending_sensor_events)
  };
};

State.prototype.applyBrainOutput = function applyBrainOutput(brainJson) {
  if (!isObject(brainJson) || typeof brainJson.action !== 'string') {
    throw new Error('Brain output must include action');
  }

  if (brainJson.action === 'SKIP') {
    this.last_brain_tick_ms = nowMs();
    return;
  }

  if (brainJson.action !== 'UPDATE') {
    throw new Error('Brain output action must be SKIP or UPDATE');
  }

  assertArray(brainJson.transcript_add, 'transcript_add');
  assertArray(brainJson.tips_add, 'tips_add');
  assertArray(brainJson.tips_remove, 'tips_remove');
  assertArray(brainJson.suggested_responses, 'suggested_responses');

  for (var i = 0; i < brainJson.transcript_add.length; i++) {
    var line = brainJson.transcript_add[i];
    if (!isObject(line) || typeof line.line_id !== 'string' || typeof line.speaker !== 'string' || typeof line.text !== 'string') {
      throw new Error('Invalid transcript_add line');
    }
    if (this._lineIdSet.has(line.line_id)) {
      continue;
    }
    this._lineIdSet.add(line.line_id);
    this.transcript_lines.push(line);
    this.committed_transcript_cursor = this.transcript_lines.length - 1;
  }

  for (var j = 0; j < brainJson.tips_add.length; j++) {
    var tip = brainJson.tips_add[j];
    if (!isObject(tip) || typeof tip.tip_id !== 'string' || typeof tip.text !== 'string') {
      throw new Error('Invalid tips_add item');
    }
    if (this._tipIdSet.has(tip.tip_id)) {
      continue;
    }
    this._tipIdSet.add(tip.tip_id);
    this.active_tips.push(tip);
  }

  for (var k = 0; k < brainJson.tips_remove.length; k++) {
    var tipId = brainJson.tips_remove[k];
    if (typeof tipId !== 'string') {
      throw new Error('Invalid tips_remove item');
    }
    if (!this._tipIdSet.has(tipId)) {
      continue;
    }
    this._tipIdSet.delete(tipId);
    this.active_tips = this.active_tips.filter(function (t) { return t.tip_id !== tipId; });
  }

  this.suggested_responses = brainJson.suggested_responses.slice();
  this.last_brain_tick_ms = nowMs();

  if (this.transcript_lines.length > this._windowSize) {
    this.transcript_lines = this.transcript_lines.slice(this.transcript_lines.length - this._windowSize);
    this._lineIdSet = new Set(this.transcript_lines.map(function (t) { return t.line_id; }));
    this.committed_transcript_cursor = this.transcript_lines.length - 1;
  }

  this.pending_sensor_events = [];
};

State.prototype.getHudFrame = function getHudFrame(tsMs) {
  var ts = Number.isFinite(tsMs) ? tsMs : nowMs();
  return {
    frame_id: this.current_frame_id,
    ts_ms: ts,
    transcript_lines: clone(this.transcript_lines),
    active_tips: clone(this.active_tips),
    suggested_responses: this.suggested_responses.slice(),
    scene_summary: this.scene_summary,
    signals: clone(this.signals)
  };
};

State.prototype.tickHud = function tickHud(nowMsValue) {
  this.current_frame_id += 1;
  return this.getHudFrame(nowMsValue);
};

State.prototype.tickBrain = function tickBrain(nowMsValue) {
  this.last_brain_tick_ms = Number.isFinite(nowMsValue) ? nowMsValue : nowMs();
};

module.exports = State;
