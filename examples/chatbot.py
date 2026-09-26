"""A chatbot memory loop without any LLM.

Each user message is scanned for facts and preferences worth remembering;
before answering, the most relevant memories are formatted for the prompt.
Replace ``fake_llm`` with a call to your model.

Run:  python examples/chatbot.py
"""

import tempfile
from pathlib import Path

from highhxpack import Memory
from highhxpack.integrations import ChatMemory

BASE_PROMPT = "You are a helpful assistant."


def fake_llm(system_prompt: str, message: str) -> str:
    lines = system_prompt.count("\n- ")
    return f"(model reply using {lines} remembered facts)"


conversation = [
    "Hi! I'm a backend developer and I prefer Go for services.",
    "My manager is Priya. What's a good way to structure a Go project?",
    "I want to learn Kubernetes this year.",
    "Actually, I prefer Rust for services now.",
    "Which language should I use for my next service?",
]

with tempfile.TemporaryDirectory() as tmp, Memory(Path(tmp) / "chat.db") as memory:
    chat = ChatMemory(memory, user_id="alice")
    for message in conversation:
        context = chat.context(message)
        system_prompt = f"{BASE_PROMPT}\n\n{context}" if context else BASE_PROMPT
        print(f"user: {message}")
        print(f"bot:  {fake_llm(system_prompt, message)}")
        stored = chat.observe(message)
        for result in stored:
            print(f"      remembered [{result.memory.memory_type}] {result.memory.content}")
        print()

    print("Context for the last question:")
    print(chat.context("Which language should I use for my next service?"))
    print("\nHistory is kept; the old preference is superseded, not deleted:")
    for record in memory.list("alice", status="superseded"):
        print(f"  superseded: {record.content}")
