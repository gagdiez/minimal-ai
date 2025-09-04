import chromadb
import openai
import os

# Set environment variable to disable the warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    chroma_client = chromadb.PersistentClient(path="./chroma/")
    collection = chroma_client.get_collection(name='cities')

    print("🤖 Agent: How can I help you today?")

    user_input = input("\n👤 User: ").strip()

    relevant_docs = collection.query(query_texts=user_input, n_results=3)

    client = openai.OpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key="<API KEY>",
    )

    response = client.chat.completions.create(
        model="accounts/fireworks/models/llama4-maverick-instruct-basic",
        messages=[
            {
             "role": "documentation",
              "content": "\n\n".join(relevant_docs['documents'][0])
            },
            {
                "role": "system",
                "content": (
                    "Use the provided documentation to answer.\n"
                    "- If the answer is not in the documentation, say 'I don't know'.\n"
                    "- Be concise and as close as possible to the documentation."
                )
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
    )

    # Handle the response
    print("🤖 Agent:", response.choices[0].message.content)


if __name__ == "__main__":
    main()
