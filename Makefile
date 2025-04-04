# Makefile

PYTHON = python3

VENV_DIR = venv


UNDERSAMPLING ?= 0.0  
TUNING_FLAG ?= False

run-pipeline:
	@echo "Running the full pipeline with TUNING_FLAG=$(TUNING_FLAG) and UNDERSAMPLING=$(UNDERSAMPLING)..."
	@$(PYTHON) ./run.py --tuning $(TUNING_FLAG) --undersampling $(UNDERSAMPLING)

venv:
	@echo "Creating virtual environment..."
	@$(PYTHON) -m venv $(VENV_DIR)
	@echo "Virtual environment created. Run 'source $(VENV_DIR)/bin/activate' to activate it."
	@$(VENV_DIR)/bin/pip install --upgrade pip
	@$(VENV_DIR)/bin/pip install -r requirements.txt

clean:
	@echo "Cleaning up generated files..."
	@rm -f data/input_data.csv
	@rm -rf __pycache__

install:
	@echo "Installing dependencies..."
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install -r requirements.txt

.PHONY: run-pipeline venv clean install