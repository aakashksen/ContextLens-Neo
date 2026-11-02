'use strict';

var path = require('path');
var fs = require('fs');
var fastifyFactory = require('fastify');

function FrameCaptureServer() {
  this.fastify = fastifyFactory({ logger: false });
  this.server = null;
  this.baseUrl = null;
  this.latestFrame = null;
  this.waiters = [];
}

FrameCaptureServer.prototype.start = async function (port) {
  var self = this;
  var htmlPath = path.join(__dirname, 'capture.html');
  var html = fs.readFileSync(htmlPath, 'utf8');

  this.fastify.get('/capture.html', function (_req, reply) {
    reply.type('text/html').send(html);
  });

  this.fastify.post('/frame', function (request, reply) {
    var body = request.body || {};
    if (!body.image_base64) {
      reply.code(400).send({ ok: false });
      return;
    }
    self.latestFrame = {
      image_base64: body.image_base64,
      ts_ms: Number(body.ts_ms) || Date.now()
    };
    while (self.waiters.length) {
      var waiter = self.waiters.shift();
      waiter.resolve(self.latestFrame);
    }
    reply.send({ ok: true });
  });

  this.fastify.get('/health', function (_req, reply) {
    reply.send({ ok: true });
  });

  await this.fastify.listen({ port: port || 0, host: '127.0.0.1' });
  this.server = this.fastify.server;
  var address = this.server.address();
  if (typeof address === 'string') {
    this.baseUrl = address;
  } else {
    this.baseUrl = 'http://127.0.0.1:' + address.port;
  }
};

FrameCaptureServer.prototype.stop = async function () {
  if (this.fastify) {
    await this.fastify.close();
  }
  this.server = null;
  this.baseUrl = null;
  this.latestFrame = null;
  this.waiters = [];
};

FrameCaptureServer.prototype.getBaseUrl = function () {
  return this.baseUrl;
};

FrameCaptureServer.prototype.waitForFrame = function (timeoutMs) {
  var self = this;
  return new Promise(function (resolve, reject) {
    var timer = null;
    var waiter = {
      resolve: function (frame) {
        if (timer) clearTimeout(timer);
        resolve(frame);
      }
    };
    self.waiters.push(waiter);
    if (Number.isFinite(timeoutMs) && timeoutMs > 0) {
      timer = setTimeout(function () {
        self.waiters = self.waiters.filter(function (w) { return w !== waiter; });
        reject(new Error('frame_timeout'));
      }, timeoutMs);
    }
  });
};

module.exports = {
  FrameCaptureServer: FrameCaptureServer
};
