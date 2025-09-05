import openai
import base64


# Helper function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# The path to your image
image_path = "./img.jpg"

# The base64 string of the image
image_base64 = encode_image(image_path)

client = openai.OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key="<API KEY>",
)

response = client.chat.completions.create(
    model="accounts/fireworks/models/llama4-maverick-instruct-basic",
    messages=[{
        "role": "system",
        "content": [{
            "type": "text",
            "text": ("Extract information form the image and return it as a"
                     "strict JSON object with fields 'title', 'x-axis',"
                     "'y-axis' and 'number-of-points"),
        }, {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            },
        }, ],
    }],
)

print(response.choices[0].message.content)
