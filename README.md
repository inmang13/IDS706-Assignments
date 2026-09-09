# IDS706-Assignments
[![Python tests](https://github.com/inmang13/IDS706-Assignments/actions/workflows/test.yml/badge.svg)](https://github.com/inmang13/IDS706-Assignments/actions/workflows/test.yml)

## Project Description

Grace's Repo for IDS 607 Data Engineering System Assignments

## Project Structure

- `src/main.py` - application code
- `tests/test_main.py` - automated tests
- `Dockerfile` - container configuration
- `Makefile` - common development commands
- `.github/workflows/test.yml` - GitHub Actions workflow
- `notebooks/basic_analysis.ipynb` - streamflow data analysis and forecasting notebook
- `notebooks/rust_vs_python_intro.ipynb` - Rust primer for Python users (requires a Rust Jupyter kernel)

## Installation

Create and activate a virtual environment if desired, then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Or use the Makefile:

```bash
make install
```

## Running the Application

Run the application with:

```bash
python src/main.py
```

Or:

```bash
make run
```

The program asks for a name and prints a welcome message.

Example:

```text
Enter your name: Ammy
Ammy, welcome to the Data Engineering course.
```

The main function, `welcome_message(name)`, accepts a name and returns a formatted welcome string.

## Running Tests

Run the tests locally with:

```bash
python -m pytest -q
```

Or:

```bash
make test
```

The GitHub Actions workflow runs the tests, builds the Docker image, and runs the test suite inside Docker for every push and pull request.

## Docker

Make sure Docker Desktop is running, then build and run the application:

```bash
make docker-build
make docker-run
```

To run the tests inside the container:

```bash
make docker-test
```

## Notebooks

`notebooks/basic_analysis.ipynb` analyzes USGS daily mean discharge data (Eno River at Hillsborough, NC, 2010-2020). Covers data cleaning, monthly/seasonal/yearly flow statistics, boxplot visualizations, and a next-day flow forecast comparing XGBoost and Random Forest models on lag/rolling/calendar features.

`notebooks/rust_vs_python_intro.ipynb` introduces Rust from a Python starting point, motivated by the Rust engines behind fast Python tools such as Polars. Covers the parts that look familiar (variables, conditions, loops), then move semantics, ownership, and borrowing, including a mutation bug Python runs happily and Rust rejects at compile time. Runs on a Rust Jupyter kernel rather than Python; a few cells are meant to fail to compile.

