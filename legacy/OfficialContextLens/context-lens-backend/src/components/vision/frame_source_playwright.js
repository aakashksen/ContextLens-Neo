'use strict';

var playwright = require('playwright');
var FrameCaptureServer = require('./server').FrameCaptureServer;

function PlaywrightFrameSource(timeoutMs) {
  this.browser = null;
  this.page = null;
  this.server = new FrameCaptureServer();
  this.timeoutMs = timeoutMs || 5000;
}

PlaywrightFrameSource.prototype.start = async function () {
  await this.server.start(0);
  var baseUrl = this.server.getBaseUrl();
  this.browser = await playwright.chromium.launch({
    headless: false,
    args: []
  });
  var context = await this.browser.newContext();
  this.page = await context.newPage();
  this.page.on('console', function (msg) {
    console.log('[capture]', msg.text());
  });
  await this.page.goto(baseUrl + '/capture.html', { waitUntil: 'domcontentloaded' });
  try {
    await this.page.waitForFunction(function () { return window.__captureReady === true; }, { timeout: this.timeoutMs });
  } catch (_) {
    var error = await this.page.evaluate(function () { return window.__captureError || null; });
    console.warn('Capture not ready; select a window/screen in the picker.', error || '');
  }
};

PlaywrightFrameSource.prototype.stop = async function () {
  if (this.page) {
    await this.page.close();
    this.page = null;
  }
  if (this.browser) {
    await this.browser.close();
    this.browser = null;
  }
  await this.server.stop();
};

PlaywrightFrameSource.prototype.captureJpegBuffer = async function () {
  if (!this.page) {
    throw new Error('Playwright page not initialized');
  }
  var framePromise = this.server.waitForFrame(this.timeoutMs);
  var ok = await this.page.evaluate(function () { return window.captureAndSend(); });
  if (!ok) {
    throw new Error('capture_not_ready');
  }
  var frame = await framePromise;
  return Buffer.from(frame.image_base64, 'base64');
};

module.exports = {
  PlaywrightFrameSource: PlaywrightFrameSource
};
