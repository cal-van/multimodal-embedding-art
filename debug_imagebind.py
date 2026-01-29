import sys
import os

print(f"Python executable: {sys.executable}")
print(f"CWD: {os.getcwd()}")
print(f"Sys path: {sys.path}")

try:
    import imagebind
    print("SUCCESS: imagebind imported")
    print(f"imagebind file: {imagebind.__file__}")
    
    from imagebind import data
    print("SUCCESS: imagebind.data imported")
    
except ImportError as e:
    print(f"FAILURE: {e}")
except Exception as e:
    print(f"ERROR: {e}")
