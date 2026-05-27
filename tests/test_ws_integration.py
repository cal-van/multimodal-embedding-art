import asyncio
import httpx
import websockets
import json


async def test_websocket():
    print("Testing WebSocket Integration...")

    # 1. Create Job
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000") as client:
        try:
            response = await client.post(
                "/jobs/", json={"target_text": ["ws_test"], "output_modality": "image"}
            )
            response.raise_for_status()
            job_id = response.json()["id"]
            print(f"Created job: {job_id}")
        except httpx.ConnectError:
            print("Failed to connect to server. Is it running?")
            return

    # 2. Connect to WS
    uri = f"ws://127.0.0.1:8000/jobs/{job_id}/ws"
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to WS: {uri}")
            async for message in websocket:
                data = json.loads(message)
                print(f"Received: {data}")

                # Exit conditions
                if data.get("type") == "init":
                    print(f"Received INIT state: status={data.get('status')}")
                    if data.get("status") in ["completed", "failed"]:
                        print("Job already finished at init!")
                        break

                if data.get("type") == "status":
                    status = data.get("status")
                    if status == "completed":
                        print("Job completed successfully!")
                        break
                    if status == "failed":
                        print("Job failed (expected if ImageBind missing)!")
                        break

                if data.get("error"):
                    print("Received error message!")
                    break
    except Exception as e:
        print(f"WS Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_websocket())
