PY ?= python3
.PHONY: test run run-small viewer
test:
	$(PY) -m pytest -q
run:            ## full experiment (needs data/training_setA, data/training_setB)
	$(PY) -m src.run --data data --results results
run-small:      ## quick smoke run on a patient subsample
	$(PY) -m src.run --data data --results results --max-patients 2000 --n-splits 3 --n-boot 50
viewer:
	$(PY) -m http.server 8000   # then open http://localhost:8000/viewer/
