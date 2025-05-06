install_uv:
	@which uv >/dev/null || (echo "Installing UV..." && curl -Ls https://astral.sh/uv/install.sh | sh)

install_dev: install_uv
	uv sync

.PHONY: sync
sync: install_uv
	uv sync

compile: install_uv
	uv pip compile --resolver=backtracking --no-emit-index-url -o requirements.txt pyproject.toml

up: compile sync

download_data: sync
	uv run src/mushroom/utils/download_data.py

run: sync download_data
	uv run src/mushroom/pipeline/pipeline.py

clean:
	@echo "Cleaning up..."
	@rm -rf __pycache__ .mypy_cache .pytest_cache .coverage .coverage.*
	@rm -rf .mypy_cache .pytest_cache .coverage .coverage.*
	@rm -rf requirements.txt
	@rm -rf data

.PHONY: install_uv install_dev compile sync up download_data run, clean