'use strict';

var fs = require('fs');
var path = require('path');
var deepgramSdk = require('@deepgram/sdk');
var GenerativeAI = require('@google/generative-ai');

function loadDotEnv() {
  var envPath = path.join(__dirname, '..', '.env');
  if (!fs.existsSync(envPath)) return;
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

async function translateAudio(filePath) {
  if (process.env.TRANSLATE !== '1') {
    throw new Error('TRANSLATE flag is not enabled');
  }

  var apiKey = process.env.DEEPGRAM_API_KEY || '';
  var geminiKey = process.env.GEMINI_API_KEY || '';
  if (!apiKey) throw new Error('DEEPGRAM_API_KEY missing');
  if (!geminiKey) throw new Error('GEMINI_API_KEY missing');

  var deepgram = deepgramSdk.createClient(apiKey);
  var genAI = new GenerativeAI.GoogleGenerativeAI(geminiKey);
  var audio = fs.readFileSync(filePath);

  var response = await deepgram.listen.prerecorded.transcribeFile(audio, {
    detect_language: true,
    model: 'nova-2',
    smart_format: true,
    paragraphs: true
  });

  if (response.error) {
    throw response.error;
  }

  var result = response.result;
  if (!result) {
    throw new Error('No result from Deepgram');
  }

  var channel = result.results.channels[0];
  var language = channel.detected_language || 'unknown';
  process.stderr.write('Detected Language: ' + language + '\n\n');

  var paragraphs = channel.alternatives[0].paragraphs;
  if (!paragraphs || !paragraphs.paragraphs) return;

  for (var p = 0; p < paragraphs.paragraphs.length; p++) {
    var para = paragraphs.paragraphs[p];
    for (var s = 0; s < para.sentences.length; s++) {
      var sentence = para.sentences[s];
      process.stdout.write('[' + sentence.start + 's - ' + sentence.end + 's]: ' + sentence.text + '\n');
      if (language !== 'en') {
        var model = genAI.getGenerativeModel({ model: 'gemini-pro' });
        var prompt = 'Translate this to English: \"' + sentence.text + '\"';
        var genResult = await model.generateContent(prompt);
        var translation = genResult.response.text();
        process.stdout.write('  -> English: ' + translation + '\n\n');
      }
    }
  }
}

var filePath = process.argv[2];
if (!filePath) {
  process.stderr.write('Please provide a file path as an argument\n');
  process.exit(1);
}

translateAudio(filePath).catch(function (err) {
  process.stderr.write('Error: ' + (err && err.message ? err.message : String(err)) + '\n');
  process.exit(1);
});
