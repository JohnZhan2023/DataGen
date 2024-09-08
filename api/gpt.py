from openai import OpenAI
client = OpenAI()
def openai_api():
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "write a haiku about ai"}
        ]
    )
    return completion.choices[0].message.content

if __name__=='__main__':
    response = openai_api()
    print(response)