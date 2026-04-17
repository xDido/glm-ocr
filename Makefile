# GLM-OCR two-container load-test harness
# Usage: make up  |  make smoke  |  make load-asyncio  |  ...

SHELL := /bin/bash
COMPOSE := docker compose
COMPOSE_MS := docker compose -f docker-compose.yml -f docker-compose.ministack.yml

CPU_URL ?= http://localhost:5002
SGL_URL ?= http://localhost:30000

.PHONY: help up down build logs ps smoke runtime runtime-full \
        load-asyncio load-locust load-k6 \
        datasets datasets-funsd datasets-omnidocbench \
        ministack-up ministack-down register-tasks \
        clean

help:
	@echo "Targets:"
	@echo "  up                - build + start CPU + GPU containers (docker compose up -d)"
	@echo "  down              - stop + remove containers"
	@echo "  build             - build images only"
	@echo "  logs              - tail compose logs"
	@echo "  ps                - show container status"
	@echo "  smoke             - run end-to-end smoke test against CPU container"
	@echo "  runtime           - terse integrity report (env vs live process/SGLang)"
	@echo "  runtime-full      - full runtime JSON including Prometheus metrics"
	@echo "  datasets          - pull FUNSD + OmniDocBench into ./datasets"
	@echo "  load-asyncio      - run aiohttp bench script"
	@echo "  load-locust       - run locust headless for 2 minutes"
	@echo "  load-k6           - run k6 scenario"
	@echo "  ministack-up      - bring up CPU + GPU + ministack sidecar"
	@echo "  ministack-down    - stop the ministack stack"
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
	bash scripts/smoke_test.sh "$(CPU_URL)"

runtime:
	CPU_URL=$(CPU_URL) MODE=summary bash scripts/runtime_probe.sh

runtime-full:
	CPU_URL=$(CPU_URL) MODE=full bash scripts/runtime_probe.sh

datasets: datasets-funsd

datasets-funsd:
	bash scripts/fetch_datasets.sh funsd

datasets-omnidocbench:
	bash scripts/fetch_datasets.sh omnidocbench

load-asyncio:
	python loadtest/asyncio/bench.py --host $(CPU_URL) --concurrency 16 --total 128

load-locust:
	cd loadtest/locust && \
	  locust -f locustfile.py --headless -u 50 -r 5 -t 2m \
	    --host $(CPU_URL) --csv ../results/locust

load-k6:
	k6 run loadtest/k6/ocr_load.js \
	  -e HOST=$(CPU_URL) \
	  --summary-export=loadtest/results/k6.json

ministack-up:
	$(COMPOSE_MS) up -d --build

ministack-down:
	$(COMPOSE_MS) down

register-tasks:
	bash aws/register-tasks.sh

clean:
	$(COMPOSE) down -v
	rm -rf loadtest/results/* || true
	touch loadtest/results/.gitkeep
