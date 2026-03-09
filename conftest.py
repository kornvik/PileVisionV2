"""Mock depthai module so tests can run without OAK-D hardware SDK."""
import sys
from unittest.mock import MagicMock

# depthai is a hardware SDK that won't be installed in test environments
sys.modules["depthai"] = MagicMock()
