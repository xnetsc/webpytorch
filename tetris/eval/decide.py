"""Answer decide() requests for eval.mjs: one JSON request per stdin line, one reply per line.

Runs the decision model through the same SDK call the page makes, on the host -- numpy, not
the GPU -- so the page's questions can be scored over hundreds of games without a browser.

    python3 tetris/eval/decide.py path/to/xDecision-Q8_0.gguf
"""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import webtorch  # noqa: E402


async def main(path):
    webtorch.use_default_io(cache=False)
    model = await webtorch.load(path)
    print(json.dumps({"ready": True, "kind": model.kind}), flush=True)
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            return
        request = json.loads(line)
        started = time.perf_counter()
        try:
            reply = model.decide(request["state"], request["questions"])
            reply["ms"] = (time.perf_counter() - started) * 1000
        except Exception as error:  # reported to the caller, which stops
            reply = {"error": repr(error)}
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
