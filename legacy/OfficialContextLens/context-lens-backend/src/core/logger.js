'use strict';

function format(level, msg, meta) {
  var line = '[' + level.toUpperCase() + '] ' + msg;
  if (meta) {
    try {
      line += ' ' + JSON.stringify(meta);
    } catch (err) {
      line += ' [meta_unserializable]';
    }
  }
  return line;
}

function info(msg, meta) {
  process.stderr.write(format('info', msg, meta) + '\n');
}

function warn(msg, meta) {
  process.stderr.write(format('warn', msg, meta) + '\n');
}

function error(msg, meta) {
  process.stderr.write(format('error', msg, meta) + '\n');
}

module.exports = {
  info: info,
  warn: warn,
  error: error
};
