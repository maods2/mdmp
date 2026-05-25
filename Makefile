.PHONY: clean

clean:
	find . -type d -name "__pycache__" \
		! -path "./env_tcc_eeg/*" \
		! -path "./.venv-win/*" \
		! -path "./venv/*" \
		-exec rm -rv {} \;
