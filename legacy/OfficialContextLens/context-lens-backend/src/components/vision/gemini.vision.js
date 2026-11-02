'use strict';

var GenerativeAI = require('@google/generative-ai');
var zod = require('zod');
var VISION_PROMPT = require('./vision.prompt').VISION_PROMPT;

var z = zod.z;

var enumValues = {
  lightingLevels: ['low', 'normal', 'bright'],
  lightingSources: ['natural', 'indoor', 'mixed'],
  noiseLevels: ['quiet', 'moderate', 'busy'],
  proximity: ['close', 'medium', 'far'],
  orientation: ['facing_camera', 'side_profile', 'back_turned', 'mixed'],
  facial: ['neutral', 'positive', 'negative', 'confused', 'engaged', 'unknown'],
  posture: ['open', 'closed', 'leaning_in', 'leaning_away', 'restless', 'unknown'],
  gaze: ['toward_camera', 'away_from_camera', 'downward', 'mixed', 'unknown'],
  interaction: ['conversation', 'presentation', 'studying', 'waiting', 'unknown']
};

function toNumber(value) {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string') {
    var parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
}

function toString(value) {
  return typeof value === 'string' ? value.trim() || undefined : undefined;
}

function normalizeEnumValue(value) {
  return value.trim().toLowerCase().replace(/[\s-]+/g, '_');
}

function isRecord(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function toEnum(allowed) {
  return function (value) {
    if (typeof value !== 'string') return undefined;
    var normalized = normalizeEnumValue(value);
    return allowed.indexOf(normalized) >= 0 ? normalized : undefined;
  };
}

function toStringArray(max) {
  return function (value) {
    if (!Array.isArray(value)) return undefined;
    var cleaned = value
      .map(function (item) { return typeof item === 'string' ? item.trim() : ''; })
      .filter(Boolean)
      .slice(0, max);
    return cleaned.length ? cleaned : undefined;
  };
}

var LightingSchema = z.object({
  level: z.preprocess(toEnum(enumValues.lightingLevels), z.enum(enumValues.lightingLevels)).optional(),
  source: z.preprocess(toEnum(enumValues.lightingSources), z.enum(enumValues.lightingSources)).optional(),
  confidence: z.preprocess(toNumber, z.number().min(0).max(1)).optional()
});

var NoiseSchema = z.object({
  level: z.preprocess(toEnum(enumValues.noiseLevels), z.enum(enumValues.noiseLevels)).optional(),
  confidence: z.preprocess(toNumber, z.number().min(0).max(1)).optional()
});

var EnvironmentSchema = z.preprocess(function (value) {
  if (typeof value === 'string') {
    return { label: value };
  }
  if (!isRecord(value)) return undefined;
  var label = value.label || value.description || value.type || value.env_label;
  var objects = value.objects || value.object_list || value.items;
  var noise = value.noise_or_busyness || value.noise || value.busyness;
  var merged = {};
  if (label) merged.label = label;
  if (objects) merged.objects = objects;
  if (noise) merged.noise_or_busyness = noise;
  return Object.assign({}, value, merged);
}, z.object({
  label: z.preprocess(toString, z.string()).optional(),
  confidence: z.preprocess(toNumber, z.number().min(0).max(1)).optional(),
  objects: z.preprocess(toStringArray(8), z.array(z.string()).max(8)).optional(),
  lighting: LightingSchema.optional(),
  noise_or_busyness: NoiseSchema.optional()
})).optional();

var PeopleSchema = z.preprocess(function (value) {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return { count_estimate: value };
  }
  if (typeof value === 'string') {
    var parsed = Number(value);
    if (Number.isFinite(parsed)) return { count_estimate: parsed };
  }
  if (!isRecord(value)) return undefined;
  var count = value.count_estimate || value.count || value.people_count || value.num;
  var orientation = value.orientation_summary || value.orientation;
  var merged = {};
  if (count !== undefined) merged.count_estimate = count;
  if (orientation) merged.orientation_summary = orientation;
  return Object.assign({}, value, merged);
}, z.object({
  count_estimate: z.preprocess(toNumber, z.number().min(0)).optional(),
  count_confidence: z.preprocess(toNumber, z.number().min(0).max(1)).optional(),
  proximity: z.preprocess(toEnum(enumValues.proximity), z.enum(enumValues.proximity)).optional(),
  orientation_summary: z.preprocess(toEnum(enumValues.orientation), z.enum(enumValues.orientation)).optional(),
  confidence: z.preprocess(toNumber, z.number().min(0).max(1)).optional()
})).optional();

var CueNotes = z.preprocess(toStringArray(2), z.array(z.string()).max(2)).optional();

function SummarySchema(allowed) {
  return z.preprocess(function (value) {
    if (typeof value === 'string') return { label: value };
    if (typeof value === 'number') return { confidence: value };
    if (!isRecord(value)) return undefined;
    return value;
  }, z.object({
    label: z.preprocess(toEnum(allowed), z.enum(allowed)).optional(),
    confidence: z.preprocess(toNumber, z.number().min(0).max(1)).optional(),
    notes: CueNotes
  })).optional();
}

var InteractionSchema = z.preprocess(function (value) {
  if (typeof value === 'string') return { label: value };
  if (typeof value === 'number') return { confidence: value };
  if (!isRecord(value)) return undefined;
  return value;
}, z.object({
  label: z.preprocess(toEnum(enumValues.interaction), z.enum(enumValues.interaction)).optional(),
  confidence: z.preprocess(toNumber, z.number().min(0).max(1)).optional()
})).optional();

var SocialCuesSchema = z.object({
  facial_expression_summary: SummarySchema(enumValues.facial),
  body_posture_summary: SummarySchema(enumValues.posture),
  gaze_summary: SummarySchema(enumValues.gaze),
  interaction_context: InteractionSchema
});

var ReliabilitySchema = z.preprocess(function (value) {
  if (typeof value === 'number') return { overall_confidence: value };
  if (!isRecord(value)) return undefined;
  var overall = value.overall_confidence || value.confidence;
  var limitations = value.limitations || value.notes;
  var merged = {};
  if (overall !== undefined) merged.overall_confidence = overall;
  if (limitations) merged.limitations = limitations;
  return Object.assign({}, value, merged);
}, z.object({
  overall_confidence: z.preprocess(toNumber, z.number().min(0).max(1)).optional(),
  limitations: z.preprocess(toStringArray(3), z.array(z.string()).max(3)).optional()
})).optional();

var RawVisionSchema = z.object({
  environment: EnvironmentSchema.optional(),
  people: PeopleSchema.optional(),
  social_cues: z.preprocess(function (value) { return isRecord(value) ? value : undefined; }, SocialCuesSchema).optional(),
  reliability: ReliabilitySchema.optional(),
  notes: z.preprocess(toStringArray(3), z.array(z.string()).max(3)).optional()
});

function GeminiVisionClient(apiKey, model) {
  this.model = model || 'gemini-1.5-flash';
  this.client = apiKey ? new GenerativeAI.GoogleGenerativeAI(apiKey) : null;
}

GeminiVisionClient.prototype.isReady = function () {
  return Boolean(this.client);
};

GeminiVisionClient.prototype.sendImage = async function (buffer) {
  if (!this.client) throw new Error('Gemini API key missing');

  var attempt = async () => {
    var model = this.client.getGenerativeModel({
      model: this.model,
      generationConfig: { responseMimeType: 'application/json', temperature: 0.2 }
    });
    var result = await model.generateContent([
      { text: VISION_PROMPT },
      { inlineData: { data: buffer.toString('base64'), mimeType: 'image/jpeg' } }
    ]);
    var text = result.response.text();
    var parsed = parseJson(text);
    return toSnapshot(parsed);
  };

  try {
    return await attempt();
  } catch (_) {
    return await attempt();
  }
};

function parseJson(text) {
  var start = text.indexOf('{');
  var end = text.lastIndexOf('}');
  if (start === -1 || end === -1 || end <= start) {
    throw new Error('Invalid JSON from Gemini');
  }
  var parsed;
  try {
    parsed = JSON.parse(text.slice(start, end + 1));
  } catch (err) {
    console.warn('Gemini JSON parse failed (truncated)', text.slice(0, 400));
    throw new Error('Invalid JSON from Gemini');
  }
  var result = RawVisionSchema.safeParse(parsed);
  if (!result.success) {
    console.warn('Gemini JSON schema mismatch', result.error.issues[0] ? result.error.issues[0].message : '');
    console.warn('Gemini JSON (truncated)', text.slice(0, 400));
    throw new Error('Invalid JSON from Gemini');
  }
  return result.data;
}

function toSnapshot(parsed) {
  var environment = parsed.environment || {};
  var lighting = environment.lighting || {};
  var noise = environment.noise_or_busyness;
  var people = parsed.people || {};
  return {
    ts_ms: Date.now(),
    environment: {
      label: environment.label || 'unknown',
      confidence: environment.confidence ?? 0.6,
      objects: environment.objects ? environment.objects.slice(0, 8) : undefined,
      lighting: {
        level: lighting.level || 'normal',
        source: lighting.source,
        confidence: lighting.confidence ?? 0.6
      },
      noise_or_busyness: noise && noise.level ? {
        level: noise.level,
        confidence: noise.confidence ?? 0.5
      } : undefined
    },
    people: {
      count_estimate: Math.max(0, Math.round(people.count_estimate ?? 0)),
      count_confidence: people.count_confidence ?? 0.5,
      proximity: people.proximity || 'medium',
      orientation_summary: people.orientation_summary,
      confidence: people.confidence ?? 0.5
    },
    social_cues: normalizeSocialCues(parsed.social_cues || {}),
    reliability: {
      overall_confidence: parsed.reliability ? (parsed.reliability.overall_confidence ?? 0.6) : 0.6,
      limitations: parsed.reliability && parsed.reliability.limitations ? parsed.reliability.limitations.slice(0, 3) : undefined
    },
    notes: parsed.notes ? parsed.notes.slice(0, 3) : undefined
  };
}

function normalizeSocialCues(cues) {
  return {
    facial_expression_summary: cues.facial_expression_summary ? {
      label: cues.facial_expression_summary.label || 'unknown',
      confidence: cues.facial_expression_summary.confidence ?? 0.3,
      notes: cues.facial_expression_summary.notes ? cues.facial_expression_summary.notes.slice(0, 2) : undefined
    } : undefined,
    body_posture_summary: cues.body_posture_summary ? {
      label: cues.body_posture_summary.label || 'unknown',
      confidence: cues.body_posture_summary.confidence ?? 0.3,
      notes: cues.body_posture_summary.notes ? cues.body_posture_summary.notes.slice(0, 2) : undefined
    } : undefined,
    gaze_summary: cues.gaze_summary ? {
      label: cues.gaze_summary.label || 'unknown',
      confidence: cues.gaze_summary.confidence ?? 0.3,
      notes: cues.gaze_summary.notes ? cues.gaze_summary.notes.slice(0, 2) : undefined
    } : undefined,
    interaction_context: cues.interaction_context ? {
      label: cues.interaction_context.label || 'unknown',
      confidence: cues.interaction_context.confidence ?? 0.3
    } : undefined
  };
}

module.exports = {
  GeminiVisionClient: GeminiVisionClient
};
