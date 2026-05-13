import urllib.request
import json
import time

url = "http://127.0.0.1:5001/api/step"
data = json.dumps({"dt": 0.016, "rho": 0.05}).encode('utf-8')

success_count = 0
for i in range(1000):
    try:
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                success_count += 1
            else:
                print(f"Error at step {i}: {response.status}")
                break
    except Exception as e:
        print(f"Exception at step {i}: {e}")
        break
    if i % 100 == 0:
        print(f"Step {i} ok")

print(f"Final success count: {success_count}/1000")
