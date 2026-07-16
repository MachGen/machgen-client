# machgen-client

External Python client for the MachGen platform.

## Installation

```bash
pip install machgen-client
```

Requires Python 3.11 or newer.

To install the latest source snapshot for development:

```bash
pip install "git+https://github.com/MachGen/machgen-client.git"
```

## Quick Start

Check the [docs](https://www.machgen.ai/docs/`) on how authentication
works and get an API key from there.

Populate the API key either in env var `MACHGEN_API_KEY`
or set it when creating the client `MachGenClient(api_key="...")`.

```python
from machgen.client import MachGenClient, TaskInput, TaskOutputType, VideoConfig

with MachGenClient() as client:
    handle = client.submit_task(
        TaskInput(
            prompt="...",
            model="Wan2.2-A14B",
            task_type="T2V",
            video_config=VideoConfig(duration_secs=5, height=720, width=1280, fps=16),
        )
    )
    result = client.wait(handle)
    print(result.task_output[TaskOutputType.VIDEO])
```

## Documentation

Full docs are published at https://www.machgen.ai/docs.

Runnable scripts for each task type live in the
[examples directory](https://github.com/MachGen/machgen-client/tree/main/examples).
