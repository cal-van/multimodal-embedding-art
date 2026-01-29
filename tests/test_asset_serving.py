import httpx
import os
import shutil

OUTPUTS_DIR = os.path.join(os.getcwd(), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def test_asset_serving():
    print("Testing Asset Serving...")
    
    # 1. Create a dummy file in outputs
    test_filename = "test_asset.txt"
    test_content = "This is a test asset."
    test_filepath = os.path.join(OUTPUTS_DIR, test_filename)
    
    with open(test_filepath, "w") as f:
        f.write(test_content)
        
    print(f"Created dummy asset: {test_filepath}")

    # 2. Try to fetch it via API
    url = f"http://127.0.0.1:8000/outputs/{test_filename}"
    try:
        response = httpx.get(url)
        if response.status_code == 200 and response.text == test_content:
            print(f"SUCCESS: Fetched asset from {url}")
        else:
            print(f"FAILURE: Status {response.status_code}, Content: {response.text}")
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        # Cleanup
        if os.path.exists(test_filepath):
            os.remove(test_filepath)

if __name__ == "__main__":
    test_asset_serving()
