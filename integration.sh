#!/bin/bash
BINDIR=$( dirname $0 )

# Create virtualenv and install dependencies if it doesn't exist.
echo 'detecting virtualenv...'
if [[ ! -d "${BINDIR}/env" ]]; then
    # Create virtualenv
    echo 'creating virtualenv'
    python3 -m venv env

    # Activate virtualenv
    echo 'activating virtualenv...'
    source env/bin/activate

    # Update pip.
    echo 'updating pip'
    pip3 install -U pip setuptools wheel

    # Install requirements.
    echo 'installing requirements...'
    pip3 install -r requirements.txt

    # Install test requirements.
    echo 'installing test requirements...'
    pip3 install -r requirements-test.txt

    # Deactivate virtualenv for idempotency.
    deactivate
else
    echo 'existing virtualenv detected...skipping...'
fi

# Activate virtualenv
echo 'activating virtualenv...'
source env/bin/activate

# Run the tests.
echo 'running integration tests....'
pytest integration/
