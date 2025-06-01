
.PHONY: help build test encode decode clean setup-wsl

DOCKER_CMD := $(shell if command -v docker.exe >/dev/null 2>&1; then echo "docker.exe"; else echo "docker"; fi)

PWD := $(shell pwd)

help:
	@echo "🎥 Framerecall H.265 Docker Helper (WSL Compatible)"
	@echo ""
	@echo "Setup:"
	@echo "  make setup-wsl    - Check WSL/Docker setup"
	@echo ""
	@echo "Commands:"
	@echo "  make build        - Build the Docker container"
	@echo "  make test         - Test that everything works"
	@echo "  make encode       - Encode chunks to H.265 video"
	@echo "  make clean        - Clean up Docker containers"
	@echo ""
	@echo "Examples:"
	@echo "  make encode INPUT=my_chunks.json OUTPUT=my_video.mkv"
	@echo ""
	@echo "Detected Docker: $(DOCKER_CMD)"

setup-wsl:
	@echo "🔍 Checking WSL + Docker Desktop setup..."
	@if grep -q Microsoft /proc/version 2>/dev/null; then \
		echo "✅ Running in WSL"; \
		if command -v docker.exe >/dev/null 2>&1 || command -v docker >/dev/null 2>&1; then \
			echo "✅ Docker available"; \
		else \
			echo "❌ Docker not found. Enable WSL integration in Docker Desktop"; \
			exit 1; \
		fi; \
	else \
		echo "✅ Running in native Linux"; \
	fi
	@echo "✅ Setup looks good!"

build: setup-wsl
	@echo "🏗️  Building Framerecall H.265 container..."
	$(DOCKER_CMD) build -f docker/Dockerfile -t framerecall-h265 docker/
	@echo "✅ Build complete!"

WIN_PATH := $(shell pwd | sed 's|/mnt/c|C:|')

test: build
	@echo "🧪 Testing container..."
	$(DOCKER_CMD) run
		-v "$(WIN_PATH)/data:/data" \
		-v "$(WIN_PATH)/docker/scripts:/scripts" \
		framerecall-h265 python3 /scripts/test_encoding.py

encode: build
	@if [ -z "$(INPUT)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "❌ Usage: make encode INPUT=file.json OUTPUT=video.mp4"; \
		exit 1; \
	fi
	@echo "🎬 Encoding $(INPUT) to $(OUTPUT)..."
	@echo "Using Windows path: $(WIN_PATH)"
	$(DOCKER_CMD) run
		-v "$(WIN_PATH)/data:/data" \
		-v "$(WIN_PATH)/docker/scripts:/scripts" \
		framerecall-h265 python3 /scripts/dockerized_encoder.py $(INPUT) $(OUTPUT)

encode-large: build
	@if [ -z "$(INPUT)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "❌ Usage: make encode-large INPUT=file.json OUTPUT=video.mp4"; \
		exit 1; \
	fi
	@echo "🚀 Large-scale encoding $(INPUT) to $(OUTPUT)..."
	@echo "   Allocating maximum resources for performance"
	$(DOCKER_CMD) run



		-v "$(WIN_PATH)/data:/data" \
		-v "$(WIN_PATH)/docker/scripts:/scripts" \
		framerecall-h265 python3 /scripts/h265_encode_optimized.py $(INPUT) $(OUTPUT)

clean:
	@echo "🧹 Cleaning up..."
	-$(DOCKER_CMD) rmi framerecall-h265
	-$(DOCKER_CMD) system prune -f

wsl-info:
	@if grep -q Microsoft /proc/version 2>/dev/null; then \
		echo "🐧 WSL Performance Info:"; \
		echo "   Cores: $$(nproc)"; \
		echo "   Memory: $$(free -m | awk 'NR==2{printf "%.1f", $$2/1024}')GB"; \
		echo "   Docker: $(DOCKER_CMD)"; \
		echo ""; \
		echo "💡 For better performance:"; \
		echo "   • Use WSL 2 (faster than WSL 1)"; \
		echo "   • Store files in WSL filesystem (/home/user/)"; \
		echo "   • Configure .wslconfig for more memory"; \
	else \
		echo "ℹ️  Not running in WSL"; \
	fi