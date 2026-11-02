'use strict';

var logger = require('../../core/logger');
var brain = require('./index');

function run() {
  var pass = 0;
  var fail = 0;

  function check(name, obj, shouldPass) {
    var result = brain.validateBrainOutput(obj);
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

  logger.info('Brain selftest complete', { pass: pass, fail: fail });
  if (fail > 0) {
    process.exitCode = 1;
  }
}

run();
