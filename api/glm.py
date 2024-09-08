import base64
from zhipuai import ZhipuAI
import os
def zhipuai_api(img_path, text):
    with open(img_path, 'rb') as img_file:
        img_base = base64.b64encode(img_file.read()).decode('utf-8')

    client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY")) # 填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4v-plus",  # 填写需要调用的模型名称
        messages=[
        {
            "role": "user",
            "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": img_base
                }
            },
            {
                "type": "text",
                "text": text
            }
            ]
        }
        ]
    )
    return response.choices[0].message.content

if __name__ == '__main__':
    response = zhipuai_api("pic/0a04b286e2dd5602.jpg", "describe the scene")
    print(response)