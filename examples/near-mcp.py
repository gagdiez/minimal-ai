import asyncio
import openai
import json

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

MODEL = "accounts/fireworks/models/qwen3-235b-a22b"
SYSTEM_PROMPT = """
You are a helpful assistant with tool access for NEAR Protocol.

Decision policy:
- If you can answer confidently without tools, answer directly.
- If you need on-chain data, external info, or execution, call a tool.
- If any required parameter is missing or ambiguous, ask a single clarifying question instead of calling a tool.

Tool-calling protocol (very important):
- When you decide to call a tool, output ONLY the tool call with the function name and JSON arguments
- After a tool result (tool message) arrives, either:
  1) Answer the user exactly what it asked, or
  2) Make another tool call if more info is required.

Examples:
Good:
user: What's the balance of alice.testnet?
<assistant decides to make a tool call>

Bad:
assistant: (account_view_account_summary(accountId="alice.testnet", networkId="testnet"))
"""


def convert_mcp_tools_to_openai(tools_result) -> list[dict]:
    """Convert MCP tools to OpenAI function format"""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": getattr(tool, 'inputSchema', {"type": "object", "properties": {}})
            }
        }
        for tool in tools_result.tools
    ]


async def handle_tool_calls(session, tool_calls, message_history, tools):
    """Process tool calls and add results to message history."""
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        # make sure the tool exists
        if not any(t['function']['name'] == tool_name for t in tools):
            return message_history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": "Error: Tool not found"
            })

        print(f"🔧 Calling tool: {tool_name} with args: {tool_args}")
        tool_result = await session.call_tool(tool_name, tool_args)

        # Use proper OpenAI tool message format
        message_history.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_name,
            "content": str(tool_result.content)
        })


async def chat_loop(session, openai_tools):
    """Main chat loop handling user input and AI responses"""
    message_history = [{"role": "system", "content": SYSTEM_PROMPT}]

    client = openai.OpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key="<API KEY>",
    )

    while True:
        try:
            # Get user input if the last message wasn't a tool response
            if not message_history or message_history[-1]['role'] != 'tool':
                user_input = input("\n👤 User: ").strip()
                if user_input.lower() in ('quit', 'exit', 'bye'):
                    break
                message_history.append({"role": "user", "content": user_input})

            # Get AI response
            response = client.chat.completions.create(
                model=MODEL,
                messages=message_history,
                tools=openai_tools,
                tool_choice="auto",
            )

            message = response.choices[0].message

            if message.tool_calls:
                # Handle tool calls
                await handle_tool_calls(
                    session, message.tool_calls,
                    message_history, tools=openai_tools
                )
            else:
                # Regular response
                message_history.append({
                    "role": "assistant",
                    "content": message.content
                })
                print(f"\n🤖 Agent: {message.content}")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            break


async def main():
    """Main application entry point."""
    print("🤖 Agent: Hi! Any questions about NEAR Protocol? (Type 'quit' to exit)")

    server_params = StdioServerParameters(
        command="npx",
        args=["@nearai/near-mcp@latest", "run"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Get and convert MCP tools
            tools_result = await session.list_tools()
            openai_tools = convert_mcp_tools_to_openai(tools_result)

            # Start chat loop
            await chat_loop(session, openai_tools)

if __name__ == "__main__":
    asyncio.run(main())
