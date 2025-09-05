import openai


def get_ai_response(client, messages, model):
    """Get response from AI model"""
    response_generator = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    full_response = ""
    for chunk in response_generator:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content

    print()  # New line after response
    return full_response


def main():
    """Main chatbot loop"""
    print("🤖 Minimal Chatbot")
    print("=================================================")

    client = openai.OpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key="<API KEY>",
    )
    conversation_history = []

    while True:
        # Get user input
        user_input = input("\n👤 User: ").strip()

        conversation_history.append({
            "role": "user",
            "content": user_input
        })

        # Get AI response
        print("🤖 Agent: ", end="")
        ai_response = get_ai_response(
            client, conversation_history,
            model="accounts/fireworks/models/llama4-maverick-instruct-basic"
        )

        conversation_history.append({
            "role": "assistant",
            "content": ai_response
        })

        # Keep conversation history manageable
        if len(conversation_history) > 20:  # 10 exchanges
            conversation_history = conversation_history[-20:]


if __name__ == "__main__":
    main()
