import requests

url = "https://api.fireworks.ai/inference/v1/workflows/accounts/fireworks/models/llama4-maverick-instruct-basic"
headers = {
    "Content-Type": "application/json",
    "Accept": "image/jpeg",
    "Authorization": "Bearer <API KEY>",
}
data = {
    "prompt": "Create an image of a beautiful sunrise over a mountain",
    "aspect_ratio": "16:9",
    "guidance_scale": 3.5,
    "num_inference_steps": 4,
    "seed": -1
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    with open("sunrise.jpg", "wb") as f:
        f.write(response.content)
    print("Image saved as sunrise.jpg")
else:
    print("Error:", response.status_code, response.text)
