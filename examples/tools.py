import openai


def validate_account(account):
    if not account.endswith('.near'):
        raise ValueError("Account must end with .near")


def get_balance(account):
    validate_account(account)
    return 1000.00  # Mock balance


def transfer_funds(from_account, to_account, amount):
    validate_account(from_account)
    validate_account(to_account)
    print(f"Transferring ${amount:.2f} from {from_account} to {to_account}")
    return {"success": True}


def main():
    print("🤖 Agent: How can I help you today?")

    user_input = input("\n👤 User: ").strip()

    # Define tools following OpenAI specifications
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_balance",
                "description": "Returns the balance of a .near account.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "account": {
                            "type": "string",
                            "description": "The account name (must end with .near)",
                        }
                    },
                    "required": ["account"],
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "transfer_funds",
                "description": "Transfers money from one account to another.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from_account": {
                            "type": "string",
                            "description": "The source account (must end with .near)",
                        },
                        "to_account": {
                            "type": "string",
                            "description": "The destination account (must end with .near)",
                        },
                        "amount": {
                            "type": "number",
                            "description": "The amount to transfer",
                        }
                    },
                    "required": ["from_account", "to_account", "amount"],
                }
            }
        }
    ]

    client = openai.OpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key="<API KEY>",
    )

    response = client.chat.completions.create(
        model="accounts/fireworks/models/llama4-maverick-instruct-basic",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an assistant on a banking app with tools for checking balances and transferring funds.\n"
                    "- If any required parameter is missing or ambiguous, ask a clarifying question instead of calling a tool.\n"
                    "- Only call a tool when you have all required parameters explicitly provided by the user."
                )
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        tools=tools,
    )

    # Handle the response
    print("🤖 Agent:", response.choices[0].message)


if __name__ == "__main__":
    main()
