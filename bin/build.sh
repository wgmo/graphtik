#!/bin/bash

# clean, or invalid files in packages
rm -vrf ./build/* ./dist/* ./*.pyc ./*.tgz ./*.egg-info
python -m build
sphinx-build -Wj auto -D graphtik_warning_is_error=true docs/source/ docs/build/