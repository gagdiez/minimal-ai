import openai


def main():
    print("🤖 Agent: Hi! Any React questions?")

    user_input = input("\n👤 User: ").strip()

    client = openai.OpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key="<API KEY>",
    )

    response = client.responses.create(
        model="accounts/fireworks/models/llama4-maverick-instruct-basic",
        input=user_input,
        tools=[{"type": "sse", "server_url": "https://gitmcp.io/reactjs/react.dev"}]
    )

    # Handle the response
    print("🤖 Agent:", response.output[-1].content[0].text)


if __name__ == "__main__":
    main()
