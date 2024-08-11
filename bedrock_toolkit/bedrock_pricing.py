"""Amazon Bedrock On-Demand Pricing"""

# Price per 1,000 input tokens
# https://aws.amazon.com/bedrock/pricing/
BEDROCK_PRICING: dict[str, dict[str, dict[str, float]]] = {
    "us-east-1": {
        "cohere.command-light-text-v14": {"input": 0.0003, "output": 0.0006},
        "cohere.command-r-plus-v1:0": {"input": 0.0030, "output": 0.0150},
        "cohere.command-r-v1:0": {"input": 0.0005, "output": 0.0015},
        "anthropic.claude-3-5-sonnet-20240620-v1:0": {"input": 0.003, "output": 0.015},
        "anthropic.claude-3-haiku-20240307-v1:0": {"input": 0.00025, "output": 0.00125},
        "anthropic.claude-3-sonnet-20240229-v1:0": {"input": 0.003, "output": 0.015},
        "mistral.mistral-large-2402-v1:0": {"input": 0.004, "output": 0.012},
        "mistral.mistral-small-2402-v1:0": {"input": 0.001, "output": 0.003},
    },
    "us-west-2": {
        "cohere.command-light-text-v14": {"input": 0.0003, "output": 0.0006},
        "cohere.command-r-plus-v1:0": {"input": 0.0030, "output": 0.0150},
        "cohere.command-r-v1:0": {"input": 0.0005, "output": 0.0015},
        "anthropic.claude-3-5-sonnet-20240620-v1:0": {"input": 0.003, "output": 0.015},
        "anthropic.claude-3-haiku-20240307-v1:0": {"input": 0.00025, "output": 0.00125},
        "anthropic.claude-3-sonnet-20240229-v1:0": {"input": 0.003, "output": 0.015},
        "mistral.mistral-large-2402-v1:0": {"input": 0.004, "output": 0.012},
        "mistral.mistral-small-2402-v1:0": {"input": 0.001, "output": 0.003},
    },
    "us-east-2": {
        "cohere.command-light-text-v14": {"input": 0.0003, "output": 0.0006},
        "cohere.command-r-plus-v1:0": {"input": 0.0030, "output": 0.0150},
        "cohere.command-r-v1:0": {"input": 0.0005, "output": 0.0015},
        "anthropic.claude-3-5-sonnet-20240620-v1:0": {"input": 0.003, "output": 0.015},
        "anthropic.claude-3-haiku-20240307-v1:0": {"input": 0.00025, "output": 0.00125},
        "anthropic.claude-3-sonnet-20240229-v1:0": {"input": 0.003, "output": 0.015},
        "mistral.mistral-large-2402-v1:0": {"input": 0.004, "output": 0.012},
        "mistral.mistral-small-2402-v1:0": {"input": 0.001, "output": 0.003},
    },
    "us-west-1": {
        "cohere.command-light-text-v14": {"input": 0.0003, "output": 0.0006},
        "cohere.command-r-plus-v1:0": {"input": 0.0030, "output": 0.0150},
        "cohere.command-r-v1:0": {"input": 0.0005, "output": 0.0015},
        "anthropic.claude-3-5-sonnet-20240620-v1:0": {"input": 0.003, "output": 0.015},
        "anthropic.claude-3-haiku-20240307-v1:0": {"input": 0.00025, "output": 0.00125},
        "anthropic.claude-3-sonnet-20240229-v1:0": {"input": 0.003, "output": 0.015},
        "mistral.mistral-large-2402-v1:0": {"input": 0.004, "output": 0.012},
        "mistral.mistral-small-2402-v1:0": {"input": 0.001, "output": 0.003},
    },
}
