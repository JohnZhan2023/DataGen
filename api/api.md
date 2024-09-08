# Collections of current LLM or VLM

## [Qwen VL](https://github.com/JohnZhan2023/DataGen.git)

| model | free tokens |
|--------|--------|
| qwen-vl-max-0809| 100 million |
| qwen-vl-max | 100 million |

qwen can't grasp where the trajectory is.
```json
{
    "Meta_Actions": [
        "speed up",
        "turn right"
    ],
    "Decision_Description": [
        {
            "Action": "speed up",
            "Subject": "ego vehicle",
            "Duration": "5 seconds"
        },
        {
            "Action": "turn right",
            "Subject": "left lane",
            "Duration": "2 seconds"
        }
    ]
}
```

## [ZHIPU](https://www.zhipuai.cn/)
```json
{
    "Meta_Actions": [
        "change lane",
        "accelerate"
    ],
    "Decision_Description": [
        {
            "Action": "change lane",
            "Subject": "lane",
            "Duration": "0s-1s"
        },
        {
            "Action": "accelerate",
            "Subject": "vehicle",
            "Duration": "1s-8s"
        }
    ]
}
```