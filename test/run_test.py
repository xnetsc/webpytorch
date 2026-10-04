import os
import pytest
import wgpy_test
from js import pythonIO

# Most backend tests do not use Chainer. Its optional local wheel is not present
# in a normal checkout, so importing it unconditionally used to abort every
# browser test before pytest could collect even a single matmul regression.
if "test_chainer" in str(pythonIO.testPath):
    import micropip
    await micropip.install('/lib/chainer-5.4.0-py3-none-any.whl')

test_dir = os.path.dirname(os.path.abspath(wgpy_test.__file__))
pytest.main([test_dir+pythonIO.testPath])
