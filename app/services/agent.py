import os
import logging
import re

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langgraph.prebuilt import create_react_agent

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def get_repo_path(repo_name: str) -> str:
    settings = get_settings()
    return os.path.join(settings.REPO_TEMP_DIR, repo_name)


def sanitize_collection_name(repo_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", repo_name)


@tool
def search_codebase(query: str, repo_name: str) -> str:
    """
    Search the repository for specific code usage,
    architectural patterns, or function definitions.
    """
    settings = get_settings()

    if not os.path.exists(settings.CHROMA_DB_DIR):
        return "Error: Database is empty. Please ingest the repository first."

    # ✅ Fixed: use settings.GOOGLE_API_KEY (consistent with rest of project)
    # ✅ Fixed: model="gemini-embedding-2-preview" (was "models/embedding-001")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        google_api_key=settings.GOOGLE_API_KEY,
    )

    vectorstore = Chroma(
        persist_directory=settings.CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name=sanitize_collection_name(repo_name),
    )

    docs = vectorstore.similarity_search(query, k=5)

    if not docs:
        return "No relevant code found."

    formatted_docs = []
    for d in docs:
        source = d.metadata.get("source", "Unknown file")
        formatted_docs.append(f"--- {source} ---\n{d.page_content}\n")

    return "\n".join(formatted_docs)


@tool
def read_file(file_path: str, repo_name: str) -> str:
    """
    Read the content of a specific file in the repository.
    """
    repo_path = get_repo_path(repo_name)
    full_path = os.path.join(repo_path, file_path)

    if not os.path.abspath(full_path).startswith(os.path.abspath(repo_path)):
        return "Error: File path is outside the repository."

    if not os.path.exists(full_path):
        return (
            f"Error: File {file_path} "
            f"does not exist in {repo_name}."
        )

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"


@tool
def analyze_code(query: str, file_path: str, repo_name: str) -> str:
    """
    Analyze a specific file for bugs,
    code quality issues and improvements.
    """
    file_content = read_file.invoke(
        {"file_path": file_path, "repo_name": repo_name}
    )

    if isinstance(file_content, str) and file_content.startswith("Error"):
        return file_content

    # ✅ Fixed: reads api_key from settings (not os.environ directly)
    settings = get_settings()
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=settings.GOOGLE_API_KEY,
        max_retries=2,
    )

    prompt = f"""
Analyze the following code for:

{query}

Code:

{file_content}

Provide:
1. Bugs
2. Code smells
3. Performance issues
4. Security concerns
5. Suggested improvements
"""

    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error analyzing code: {str(e)}"


@tool
def generate_docs(directory_path: str, repo_name: str) -> str:
    """
    Generate high-level documentation
    for files inside a directory.
    """
    repo_path = get_repo_path(repo_name)

    target_dir = (
        os.path.join(repo_path, directory_path)
        if directory_path
        else repo_path
    )

    if not os.path.exists(target_dir):
        return f"Error: Directory {directory_path} does not exist."

    file_summaries = []
    for root, dirs, files in os.walk(target_dir):
        if ".git" in root or "node_modules" in root:
            continue
        for file in files:
            file_path = os.path.relpath(
                os.path.join(root, file),
                repo_path,
            )
            file_summaries.append(f"- {file_path}")

    summary_text = "\n".join(file_summaries)

    # ✅ Fixed: reads api_key from settings (not os.environ directly)
    settings = get_settings()
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=settings.GOOGLE_API_KEY,
        max_retries=2,
    )

    prompt = f"""
Generate detailed documentation
for the following repository files.

Files:

{summary_text}

Include:
1. Purpose
2. Architecture overview
3. Important modules
4. Suggestions
"""

    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error generating docs: {str(e)}"


SYSTEM_PROMPT = """
You are an expert GitHub AI Code Review Agent.

Capabilities:
- Search repositories
- Read files
- Analyze code
- Generate documentation

Rules:

1. Only use repository tools when repository information is required to answer the user's question.

2. If the user greets you (e.g. "Hi", "Hello", "Hey"), respond naturally and do not analyze the repository.

3. If the user asks a general programming question that does not require repository context, answer it directly without using repository tools.

4. If the answer depends on repository content, always use the appropriate tools before responding.

5. Never invent files, functions, classes, APIs, architecture, or code that you have not retrieved from the repository.

6. If repository information cannot be found, clearly state that the information is unavailable rather than guessing.

7. When using tools, always pass the provided repo_name.

8. Base code reviews, bug reports, and architectural explanations only on retrieved repository content.

9. If multiple tool calls are needed, gather sufficient evidence before producing a final answer.

10. Be concise for simple questions and detailed for code reviews, debugging, and architectural analysis.

The repository name will be included in the user request.
When calling tools, always pass the repo_name.

Your primary goal is to provide accurate, evidence-based answers while minimizing hallucinations.
"""


class AIReviewAgent:
    # ✅ Fixed: added api_key param to match main.py calling AIReviewAgent(api_key=...)
    def __init__(self, api_key: str):
        self.api_key = api_key

        # ✅ Fixed: LLM now uses the passed-in api_key, not os.environ
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            google_api_key=self.api_key,
            max_retries=2,
        )

        self.tools = [
            search_codebase,
            read_file,
            analyze_code,
            generate_docs,
        ]

        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=SYSTEM_PROMPT,
        )

    async def process_message(
        self,
        user_message: str,
        repo_name: str,
    ) -> str:

        try:
            input_text = f"""
Repository Name: {repo_name}

User Request:
{user_message}

IMPORTANT:
Whenever you call a tool, use:
repo_name="{repo_name}"
"""

            response = await self.agent.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": input_text,
                        }
                    ]
                }
            )

            messages = response.get("messages", [])

            if not messages:
                return "No response generated."

            last_message = messages[-1]
            content = last_message.content
            
            if isinstance(content, list):
                text_parts = [
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                ]
                return "".join(text_parts)
            
            return str(content)

        except Exception as e:
            logger.error(f"Agent error: {str(e)}")
            raise