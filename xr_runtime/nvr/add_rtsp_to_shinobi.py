import sys
import json
import time
import requests
import os

def load_env(file_path=".env"):
    env_vars = {}
    try:
        with open(file_path, "r") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    env_vars[key] = value
    except FileNotFoundError:
        print(f"Error: {file_path} file not found.")
        sys.exit(1)
    return env_vars

env = load_env()

# Load environment variables from .env
GROUP_KEY = env.get("GROUP_KEY")
API_KEY = env.get("API_KEY")

if len(sys.argv) < 6:
    print("Usage: python add_rtsp_to_shinobi.py NUM_CAMERAS STREAMING_METHOD DEFAULT_MODE DEFAULT_RECORDING_TIME DEFAULT_FRAMERATE")
    sys.exit(1)

NUM_CAMERAS = int(sys.argv[1])
STREAMING_METHOD = sys.argv[2].lower()
DEFAULT_MODE = sys.argv[3].lower()
DEFAULT_RECORDING_TIME = int(sys.argv[4])
DEFAULT_FRAMERATE = sys.argv[5]

if STREAMING_METHOD not in ["gstreamer", "mediamtx"]:
    print("STREAMING_METHOD must be 'gstreamer' or 'mediamtx'")
    sys.exit(1)

# Load template.json
script_dir = os.path.dirname(os.path.abspath(__file__))
template_path = os.path.join(script_dir, "template.json")

try:
    with open(template_path, 'r') as f:
        template = json.load(f)
except FileNotFoundError:
    print(f"Error: template.json not found at {template_path}")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: Invalid JSON in template.json")
    sys.exit(1)

table = []

print("\n========== Adding the camera (wait)... ==========")

for i in range(1, NUM_CAMERAS + 1):
    index = f"{i:04d}"
    CAM_ID = f"cam{i}"
    HOST_PORT = 5049 + i
    
    if STREAMING_METHOD == "gstreamer":
        SERVICE_NAME = f"rtsp-server-{i}"
        HOST = "localhost"
        PORT = 8554
        PATH = "/xr_stream"
        RTSP_PATH = f"rtsp://localhost:{HOST_PORT}/xr_stream"
    else:
        SERVICE_NAME = os.getenv("MEDIAMTX_SERVICE_NAME")
        HOST = "localhost"
        PORT = 8554
        PATH = f"/NB_{index}_TX_CAM_RGB_MIC_p6S"
        RTSP_PATH = f"rtsp://localhost:8554/NB_{index}_TX_CAM_RGB_MIC_p6S"

    NAME = f"{CAM_ID} / Port {HOST_PORT}"

    # Create monitor config from template
    monitor_json = template.copy()
    
    # Set mode based on DEFAULT_MODE
    if DEFAULT_MODE == "record":
        mode_value = "record"
    else:
        mode_value = "start"

    # Update the main fields
    monitor_json.update({
        "mode": mode_value,
        "mid": CAM_ID,
        "name": NAME,
        "host": SERVICE_NAME,
        "port": PORT,
        "path": PATH
    })
    
    # Update the details section
    monitor_json["details"]["auto_host"] = f"rtsp://{SERVICE_NAME}:{PORT}{PATH}"
    monitor_json["details"]["cutoff"] = str(DEFAULT_RECORDING_TIME)

    monitor_json["details"]["cust_stream"] = f"-sc_threshold 0 -g {DEFAULT_FRAMERATE} -keyint_min {DEFAULT_FRAMERATE} -force_key_frames expr:gte(t,n_forced*1) -hls_flags independent_segments+delete_segments"
    monitor_json["details"]["sfps"] = DEFAULT_FRAMERATE
    monitor_json["details"]["stream_fps"] = DEFAULT_FRAMERATE
    
    # Convert details to JSON string as required by Shinobi API
    monitor_json["details"] = json.dumps(monitor_json["details"])

    url = f"http://localhost:8088/{API_KEY}/configureMonitor/{GROUP_KEY}/{CAM_ID}/"
    response = ""
    tries = 0
    while not response:
        try:
            r = requests.get(url, params={"data": json.dumps(monitor_json)})
            response = r.text.strip()
            tries += 1
            if not response:
                print(f"Waiting for Shinobi to accept {CAM_ID}... (try {tries})")
                time.sleep(1)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)
    # For mediamtx, also provide audio mount
    if STREAMING_METHOD == 'gstreamer':
        AUDIO_PATH = 'N/A'
    else:
        AUDIO_PATH = f"rtsp://localhost:8554/cam{i}"  # Now points to merged stream
    table.append((HOST_PORT, NAME, RTSP_PATH, AUDIO_PATH))

# Print the correspondence table
print("\n" + "#" * 80)
print("Glasses app port   | Name in Shinobi UI   | RTSP path (Video+Audio)")
print("-" * 80)
for port, name, rtsp, audio in table:
    print(f"{str(port):<18} | {name:<19} | {rtsp}")
print("#" * 80)