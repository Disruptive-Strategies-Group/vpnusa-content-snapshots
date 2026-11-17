#!/usr/bin/env python3
"""Fail fast if the installed boto3/botocore rejects Converse cachePoint blocks."""
import sys

from botocore.session import get_session
from botocore.validate import validate_parameters

shape = get_session().get_service_model("bedrock-runtime").operation_model("Converse").input_shape
cp = {"cachePoint": {"type": "default"}}
req = {
    "modelId": "m",
    "system": [{"text": "s"}, cp],
    "messages": [{"role": "user", "content": [{"text": "u"}, cp]}],
    "toolConfig": {"tools": [{"toolSpec": {"name": "t", "inputSchema": {"json": {}}}}, cp]},
}
try:
    validate_parameters(req, shape)
except Exception as e:
    print(f"::error::installed boto3/botocore rejects Converse cachePoint: {e}")
    sys.exit(1)
