"""Tests for production SSH transport command construction."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from llm_serving.schemas import HostConfig
from llm_serving.transport import SSHTransport


def test_sync_extract_does_not_restore_local_file_ownership(tmp_path):
    host = HostConfig.model_validate({
        "ssh_target": "gpu.example",
        "remote_root": "/workspace/staging",
        "gpu_ids": [0],
        "expected_gpu_model": "RTX",
        "hf_cache_path": "/workspace/cache/hf",
        "vllm_cache_path": "/workspace/cache/vllm",
    })
    transport = SSHTransport(host)
    process = MagicMock()
    process.communicate.return_value = (b"", b"")
    process.returncode = 0

    with patch.object(transport, "run_cmd", return_value=MagicMock(ok=True)), patch(
        "llm_serving.transport.subprocess.Popen", return_value=process
    ) as popen:
        transport.sync_directory_to_remote(tmp_path, "/workspace/staging/run")

    command = popen.call_args.args[0]
    assert "tar --no-same-owner -xzf - -C /workspace/staging/run" in command
