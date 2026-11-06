import requests

url_info = "http://127.0.0.1:5000/audio_info"
url_gen = "http://127.0.0.1:5000/generate_voice"

ref_audio = "cleaned_ratheesh.wav"

print("1. Testing /audio_info...")
with open(ref_audio, "rb") as f:
    res = requests.post(url_info, files={"audio": f})
print("Audio info status:", res.status_code)
print("Audio info response:", res.json())

print("\n2. Testing /generate_voice...")
with open(ref_audio, "rb") as f:
    data = {"text": "Hello! This is a high quality test of the upgraded voice cloning system.", "language": "en"}
    res = requests.post(url_gen, data=data, files={"audio": f})

print("Generate voice status:", res.status_code)
if res.status_code == 200:
    with open("outputs/test_output.wav", "wb") as out:
        out.write(res.content)
    print("Saved test output to outputs/test_output.wav! Size:", len(res.content), "bytes")
else:
    print("Error:", res.text)
