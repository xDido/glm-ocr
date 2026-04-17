# GLM-OCR two-container load-test harness
# Usage: make up  |  make smoke  |  make load-asyncio  |  ...

ifeq ($(OS),Windows_NT)
SHELL := C:/PROGRA~1/Git/bin/bash.exe
else
SHELL := /bin/bash
endif
.SHELLFLAGS := -c
COMPOSE := docker compose

CPU_URL ?= http://localhost:5002
SGL_URL ?= http://localhost:30000

.PHONY: help up down build logs ps smoke runtime runtime-full \
        omnidoc-asyncio omnidoc-locust omnidoc-k6 \
        obs-open \
        datasets datasets-funsd datasets-omnidocbench \
        register-tasks \
        clean

help:
	@echo "Targets:"
	@echo "  up                - build + start CPU + GPU + Prometheus + Grafana (docker compose up -d)"
	@echo "  down              - stop + remove containers"
	@echo "  build             - build images only"
	@echo "  logs              - tail compose logs"
	@echo "  ps                - show container status"
	@echo "  smoke             - run end-to-end smoke test against CPU container"
	@echo "  runtime           - terse integrity report (env vs live process/SGLang)"
	@echo "  runtime-full      - full runtime JSON including Prometheus metrics"
	@echo "  datasets          - pull FUNSD + OmniDocBench into ./datasets"
	@echo "  omnidoc-asyncio   - OmniDocBench asyncio load test (sampled pool)"
	@echo "  omnidoc-locust    - OmniDocBench locust load test (sampled pool)"
	@echo "  omnidoc-k6        - OmniDocBench k6 load test (sampled pool)"
	@echo "  obs-open          - print Prometheus + Grafana URLs"
	@echo "  register-tasks    - register ECS task defs against \$$ECS_ENDPOINT"
	@echo "  clean             - prune containers, volumes, loadtest results"

up:
	$(COMPOSE) up -d --build --pull missing

down:
	$(COMPOSE) down

build:
	$(COMPOSE) build

logs:
	$(COMPOSE) logs -f --tail=200

ps:
	$(COMPOSE) ps

smoke:
	"$(SHELL)" scripts/smoke_test.sh "$(CPU_URL)"

runtime:
	CPU_URL=$(CPU_URL) MODE=summary "$(SHELL)" scripts/runtime_probe.sh

runtime-full:
	CPU_URL=$(CPU_URL) MODE=full "$(SHELL)" scripts/runtime_probe.sh

datasets: datasets-funsd

datasets-funsd:
	"$(SHELL)" scripts/fetch_datasets.sh funsd

datasets-omnidocbench:
	"$(SHELL)" scripts/fetch_datasets.sh omnidocbench

obs-open:
	@echo "Prometheus : http://localhost:$${PROMETHEUS_PORT:-9090}"
	@echo "Grafana    : http://localhost:$${GRAFANA_PORT:-3000}"
	@echo "Dashboard  : http://localhost:$${GRAFANA_PORT:-3000}/d/glmocr-load/glm-ocr-load-test"

omnidoc-asyncio:
	CPU_URL=$(CPU_URL) "$(SHELL)" scripts/omnidoc_asyncio.sh

omnidoc-locust:
	CPU_URL=$(CPU_URL) "$(SHELL)" scripts/omnidoc_locust.sh

omnidoc-k6:
	CPU_URL=$(CPU_URL) "$(SHELL)" scripts/omnidoc_k6.sh

register-tasks:
	"$(SHELL)" aws/register-tasks.sh

clean:
	$(COMPOSE) down -v
	rm -rf loadtest/results/* || true
	touch loadtest/results/.gitkeep
