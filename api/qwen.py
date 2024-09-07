from openai import OpenAI
import os
import base64
import json
import sys
sys.path.append("/home/zhanjh/workspace/DataGen/")
from prompt.SceneDescription import SceneDescription
from prompt.SceneAnalysis import SceneAnalysis

#  base 64 format
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def qwen_api(image_path,text, model="qwen-vl-max-0809"):
    base64_image = encode_image(image_path)
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model = model,
        messages=[
            {
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": text
                },
                {
                  "type": "image_url",
                  "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                  }
                }
              ]
            }
          ]
        )
    result = completion.model_dump_json()
    result = json.loads(result)
    return result["choices"][0]["message"]["content"]

if __name__=='__main__':
    response = qwen_api("pic/0a04b286e2dd5602.jpg",SceneAnalysis())
    print(response)