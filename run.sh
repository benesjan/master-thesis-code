#!/usr/bin/env bash

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $PROJECT_ROOT

source $PROJECT_ROOT/venv/bin/activate

export PYTHONPATH="${PYTHONPATH}:$PROJECT_ROOT"

python $1