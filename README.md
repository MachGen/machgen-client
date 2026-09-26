# machgen-client

Official Python client for the MachGen platform, with task submission, polling,
and pushed status updates over server-sent events (SSE).

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

Check the [docs](https://www.machgen.ai/docs) on how authentication
works and get an API key from there.

Populate the API key either in env var `MACHGEN_API_KEY`
or set it when creating the client `MachGenClient(api_key="...")`.

```python
from machgen.client import MachGenClient, TaskInput, TaskOutputType, VideoConfig

with MachGenClient() as client:
    handle = client.submit_task(
        TaskInput(
            prompt="...",
            model="MiniMax-H3",
            task_type="T2V",
            video_config=VideoConfig(
                duration_secs=5, height=768, aspect_ratio="16:9", fps=24
            ),
        ),
        on_update=lambda state: print(state.status),
    )
    result = client.wait(handle)
    print(result.task_output[TaskOutputType.VIDEO])
```

`submit_task` returns a handle immediately. With `on_update`, the client opens a
background SSE connection and calls your callback as status updates arrive.
`wait` uses the same stream and blocks until the task completes or fails; it
also opens a stream when no callback is supplied. Set `timeout` on `wait` to
limit how long it blocks (300 seconds by default).

For manual polling, call `get_task_state(handle)` to fetch the current status
once, then repeat as needed.

## Documentation

Full docs are published at https://www.machgen.ai/docs.

Runnable scripts for each task type live in the
[examples directory](https://github.com/MachGen/machgen-client/tree/main/examples).
