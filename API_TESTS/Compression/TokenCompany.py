import os
import tokenc
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# Initialize TokenCompany client for compression
token_client = tokenc.TokenClient(
    api_key=os.getenv("TOKENCOMPANY")
)

# Initialize Gemini via LangCha
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",  # or "gemini-1.5-pro" for better results
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)


def compress_and_query(prompt: str, system_message: str = None, aggressiveness: float = 0.5):
    # Step 1: Compress the prompt using Bear-1
    compressed_response = token_client.compress_input(
        input=prompt,
        aggressiveness=aggressiveness,
    )

    compressed_prompt = compressed_response.output

    print(f"Original length: {len(prompt)} chars")
    print(f"Compressed length: {len(compressed_prompt)} chars")
    print(f"Compression ratio: {len(compressed_prompt)/len(prompt):.2%}")
    print(f"\nCompressed prompt: {compressed_prompt}\n")

    # Step 2: Send to Gemini via LangChain
    messages = []
    if system_message:
        messages.append(SystemMessage(content=system_message))
    messages.append(HumanMessage(content=compressed_prompt))

    response = llm.invoke(messages)

    return response.content


def compare_responses(prompt: str, system_message: str = None, aggressiveness: float = 0.5):
    """
    Compare Gemini responses from original vs compressed prompts side by side.
    """
    print("=" * 100)
    print("COMPARISON: Original vs Compressed Prompt")
    print("=" * 100)

    # Step 1: Compress the prompt
    compressed_response = token_client.compress_input(
        input=prompt,
        aggressiveness=aggressiveness,
    )
    compressed_prompt = compressed_response.output

    # Step 2: Send ORIGINAL prompt to Gemini
    print("\n[1/2] Querying Gemini with ORIGINAL prompt...")
    messages_original = []
    if system_message:
        messages_original.append(SystemMessage(content=system_message))
    messages_original.append(HumanMessage(content=prompt))

    response_original = llm.invoke(messages_original)

    # Step 3: Send COMPRESSED prompt to Gemini
    print("[2/2] Querying Gemini with COMPRESSED prompt...")
    messages_compressed = []
    if system_message:
        messages_compressed.append(SystemMessage(content=system_message))
    messages_compressed.append(HumanMessage(content=compressed_prompt))

    response_compressed = llm.invoke(messages_compressed)

    # Step 4: Display side by side
    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)

    # Split responses into lines for side-by-side display
    original_lines = response_original.content.split('\n')
    compressed_lines = response_compressed.content.split('\n')
    max_lines = max(len(original_lines), len(compressed_lines))

    # Pad shorter response with empty lines
    original_lines += [''] * (max_lines - len(original_lines))
    compressed_lines += [''] * (max_lines - len(compressed_lines))

    # Print header
    print(f"{'ORIGINAL PROMPT RESPONSE':<48} | {'COMPRESSED PROMPT RESPONSE':<48}")
    print("-" * 100)

    # Print side by side
    for orig, comp in zip(original_lines, compressed_lines):
        # Truncate lines if too long for display
        orig_display = (orig[:45] + '...') if len(orig) > 48 else orig
        comp_display = (comp[:45] + '...') if len(comp) > 48 else comp
        print(f"{orig_display:<48} | {comp_display:<48}")

    # Step 5: Show compression statistics
    print("\n" + "=" * 100)
    print("COMPRESSION STATISTICS")
    print("=" * 100)
    print(f"Original prompt length:    {len(prompt):,} characters")
    print(f"Compressed prompt length:  {len(compressed_prompt):,} characters")
    print(f"Compression ratio:         {len(compressed_prompt)/len(prompt):.2%}")
    print(f"Space saved:               {len(prompt) - len(compressed_prompt):,} characters ({(1 - len(compressed_prompt)/len(prompt)):.2%})")
    print(f"Aggressiveness level:      {aggressiveness}")
    print("=" * 100)


# Example usage
if __name__ == "__main__":
    test_prompt = "Explain the process of photosynthesis in detail, including the light-dependent reactions that occur in the thylakoid membranes, the Calvin cycle that takes place in the stroma, and how these processes work together to convert carbon dioxide and water into glucose and oxygen. Also discuss the role of chlorophyll and other pigments in capturing light energy, and explain how environmental factors like light intensity, temperature, and carbon dioxide concentration affect the rate of photosynthesis."

    # Compare original vs compressed responses side by side
    compare_responses(
        prompt=test_prompt,
        system_message="You are a helpful biology teacher.",
        aggressiveness=0.5
    )