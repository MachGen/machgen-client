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
            model="Wan2.2-A14B",
            task_type="T2V",
            video_config=VideoConfig(duration_secs=5, height=720, width=1280, fps=16),
        )
    )
    result = client.wait(handle)
    print(result.task_output[TaskOutputType.VIDEO])
```

To save part of a completed video without paid regeneration:

```python
with MachGenClient() as client:
    handle = client.extract_video_clip("<video-task-id>", 1.250, 3.750)
    result = client.wait(handle)
```

Real-time models use the separate LiveSession resource. Create the session over
HTTP, join the returned RTC channel with the SDK for your client platform, then
keep the protected control connection open for the duration of the call:

```python
from machgen.client import MachGenClient

with MachGenClient() as client:
    session = client.live.create(
        model="Vidu-S1",
        call_mode="video",
        max_session_seconds=60,
        avatar={
            "image_url": "https://example.com/avatar.jpg",
            "persona": "A friendly product specialist",
        },
    )
    with client.live.control(session) as control:
        control.wait_until_live()
        control.end()
```

## Documentation

Full docs are published at https://www.machgen.ai/docs.

Runnable scripts for each task type live in the
[examples directory](https://github.com/MachGen/machgen-client/tree/main/examples).
The Vidu S1 example is `examples/vidu_s1_live.py`; it requires a separate
AliRTC media client and asks before opening the billable control connection.
