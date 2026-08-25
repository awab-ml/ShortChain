# Collect

Instrument an agent with OpenLLMetry and run the ShortChain receiver.

```python
# pip install "shortchain[sdk,receiver]"
import os
from shortchain.sdk import ShortChain

ShortChain.init(
    api_key=os.environ["SHORTCHAIN_API_KEY"],
    app_name="support-agent",
    endpoint=os.environ.get("SHORTCHAIN_ENDPOINT", "http://127.0.0.1:4318"),
)

# Your existing agent code keeps working — instrumentors are enabled
# automatically. Mark each request's task root so success is labelled:
ShortChain.set_task(task_id="req-1", intent="Refund order 9921", app_name="support-agent")
result = agent.run("Refund order 9921")
ShortChain.end_task(success=True)
```

Run the receiver (single worker, writes `data/runtime/`):

```bash
shortchain receive --config configs/runtime.yaml
```

The receiver writes projected trajectories to `data/runtime/trajectories.jsonl`
(mode `0600`) and a tool catalog to `data/runtime/catalog.json`. Feed them to
the `train/` demo to build a model.