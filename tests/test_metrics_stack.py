"""Metrics are off by default and opt-in through the `obs` compose profile. The real test brings that profile up
and sees a metric served by this process arrive in Prometheus. Needs Docker; nothing is left running."""

import json
import os
import secrets
import socket
import subprocess
import time
import urllib.parse
import urllib.request
import uuid
from contextlib import closing
from pathlib import Path

import pytest
import yaml
from prometheus_client import CollectorRegistry

from contextlens.api import production
from contextlens.obs.tracing import Telemetry

ROOT = Path(__file__).resolve().parent.parent
METRICS_PORT = 9464  # what deploy/prometheus.yml scrapes


def test_metrics_are_off_unless_a_port_is_chosen(monkeypatch):
    monkeypatch.delenv("CONTEXTLENS_METRICS_PORT", raising=False)
    started = []
    monkeypatch.setattr(production, "serve_metrics", lambda *a: started.append(a))
    tele = Telemetry(registry=CollectorRegistry())
    production.serve_metrics_if_asked(tele)
    assert started == []
    monkeypatch.setenv("CONTEXTLENS_METRICS_PORT", "9999")
    production.serve_metrics_if_asked(tele)
    assert started == [(tele, 9999, "127.0.0.1")]  # the loopback unless the person names a host


def test_the_obs_services_are_only_in_the_obs_profile_and_on_the_loopback():
    services = yaml.safe_load((ROOT / "docker-compose.yml").read_text("utf-8"))["services"]
    assert services["prometheus"]["profiles"] == services["grafana"]["profiles"] == ["obs"]
    assert "obs" not in {p for n in ("db", "speech") for p in services[n].get("profiles", [])}
    scrape = yaml.safe_load((ROOT / "deploy" / "prometheus.yml").read_text("utf-8"))
    assert scrape["scrape_configs"][0]["static_configs"][0]["targets"] == [f"host.docker.internal:{METRICS_PORT}"]


def free(port: int) -> bool:
    with closing(socket.socket()) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.load(r)


@pytest.mark.real
def test_a_metric_reaches_prometheus_through_the_obs_profile(monkeypatch):
    assert free(9090) and free(3000) and free(METRICS_PORT), "9090, 3000 or 9464 is already in use"
    project = f"ctxlens-obs-{uuid.uuid4().hex[:8]}"
    env = {**os.environ, "COMPOSE_PROJECT_NAME": project, "GRAFANA_ADMIN_PASSWORD": secrets.token_urlsafe(16)}
    monkeypatch.setenv("CONTEXTLENS_METRICS_PORT", str(METRICS_PORT))
    tele = Telemetry(registry=CollectorRegistry())
    tele.record_warmup("probe", 1.5)  # a gauge with a value no other process could have produced
    production.serve_metrics_if_asked(tele)

    def compose(*args):
        return subprocess.run(["docker", "compose", "--profile", "obs", *args], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=300)  # fmt: skip

    try:
        up = compose("up", "-d", "prometheus", "grafana")
        assert up.returncode == 0, up.stderr[-800:]
        query = "http://127.0.0.1:9090/api/v1/query?" + urllib.parse.urlencode(
            {"query": 'contextlens_warmup_seconds{model="probe"}'}
        )
        end, last = time.monotonic() + 90, None
        while time.monotonic() < end:  # the scrape interval is 15 s
            try:
                last = get(query)["data"]["result"]
            except OSError:
                last = "prometheus is not answering yet"
            if last and last != "prometheus is not answering yet":
                break
            time.sleep(3)
        assert last and last[0]["value"][1] == "1.5", last
        for _ in range(30):  # Grafana started with the throwaway password and is reachable on the loopback
            try:
                assert get("http://127.0.0.1:3000/api/health")["database"] == "ok"
                break
            except OSError:
                time.sleep(2)
        else:
            raise AssertionError("Grafana did not answer on 127.0.0.1:3000: " + compose("logs", "grafana").stdout[-500:])
    finally:
        down = compose("down")  # no -v: it removes this project's containers and network, never a volume
        assert down.returncode == 0, down.stderr[-500:]
