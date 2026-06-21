import os
import asyncio
import sys

from dotenv import load_dotenv

from app.services.ingestion import ingest_repository
from app.services.agent import AIReviewAgent
from app.core.config import get_settings

load_dotenv()


async def main():

    # ✅ Validate API key before doing anything else
    settings = get_settings()

    if not settings.GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY is not set in your .env file.")
        print("Add this line to .env:  GOOGLE_API_KEY=your_key_here")
        print("Get a free key at: https://aistudio.google.com/apikey")
        sys.exit(1)

    print("=== Welcome to AI Code Review Agent (Terminal Mode) ===\n")

    repo_url = input(
        "Please enter a GitHub repository URL to analyze:\n> "
    ).strip()

    if not repo_url:
        print("No URL provided. Exiting.")
        sys.exit(0)

    print(
        f"\n[1/2] Cloning and ingesting codebase: {repo_url}..."
    )
    print(
        "This may take a minute depending on repository size."
    )

    try:
        repo_name = ingest_repository(repo_url)

        if not repo_name:
            print(
                "Failed to ingest repository or repository is empty."
            )
            sys.exit(1)

        print(
            f"Success! {repo_name} ingested into local database.\n"
        )

    except Exception as e:
        print(f"\nError ingesting repository: {e}")
        sys.exit(1)

    # ✅ Fixed: was "Starting Ollama Agent" — now correctly says Gemini
    print(f"[2/2] Starting Gemini Agent for {repo_name}...")

    try:
        # ✅ Fixed: AIReviewAgent now requires api_key argument
        agent = AIReviewAgent(api_key=settings.GOOGLE_API_KEY)

    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        sys.exit(1)

    print("\n--- Agent Ready! ---")
    print(
        "Ask questions about the codebase, architecture, "
        "bugs, code quality, or documentation."
    )
    print(
        "(Type 'exit' or 'quit' to close the terminal)\n"
    )

    while True:
        try:
            user_msg = input("You: ")

            if user_msg.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            if not user_msg.strip():
                continue

            print("\nAgent is thinking...\n")

            response = await agent.process_message(
                user_msg,
                repo_name,
            )

            print(f"Agent:\n{response}\n")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())