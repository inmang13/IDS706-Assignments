.PHONY: install test format format-check lint run docker-build docker-run docker-test clean

IMAGE_NAME := stream_prediction

# Install dependencies
install:
	python -m pip install -r requirements.txt

# Run tests
test:
	python -m pytest -q

# Format Python source and tests
format:
	python -m black src tests

# Verify formatting without changing files
format-check:
	python -m black --check src tests

# Run static style checks
lint:
	python -m flake8 src tests

# Run the application
run:
	python src/main.py

# Build the Docker image
docker-build:
	docker build -t $(IMAGE_NAME) .

# Run the application inside Docker
docker-run:
	docker run -it --rm $(IMAGE_NAME)


# Run the test suite inside Docker
docker-test:
	docker run --rm $(IMAGE_NAME) python -m pytest -q

# Clean generated files
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache