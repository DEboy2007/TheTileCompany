import os
import time
import tokenc
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import matplotlib.pyplot as plt

load_dotenv()

# Initialize clients
token_client = tokenc.TokenClient(api_key=os.getenv("TOKENCOMPANY"))
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)

# Test prompts
TEST_PROMPTS = [
    "Explain quantum entanglement and its implications for quantum computing, including how qubits can be entangled and how this property enables quantum algorithms to solve certain problems exponentially faster than classical computers.",

    "Describe the complete process of protein synthesis in eukaryotic cells, from DNA transcription in the nucleus through mRNA processing, ribosomal translation, and post-translational modifications.",

    "What are the key differences between supervised, unsupervised, and reinforcement learning in machine learning? Provide examples of use cases for each approach and explain when you would choose one over the others.",

    "Analyze the causes and consequences of the 2008 financial crisis, including the role of subprime mortgages, credit default swaps, regulatory failures, and the subsequent global economic impact.",

    "Explain how blockchain technology works, including consensus mechanisms like Proof of Work and Proof of Stake, and discuss the advantages and limitations of using blockchain for various applications beyond cryptocurrency.",

    "Describe the water cycle in detail, including evaporation, condensation, precipitation, infiltration, runoff, and transpiration, and explain how climate change is affecting these processes globally.",

    "What is the difference between SQL and NoSQL databases? Discuss their respective strengths, weaknesses, and ideal use cases, including considerations for scalability, consistency, and data modeling.",

    "Explain the concept of neuroplasticity and how the brain reorganizes itself by forming new neural connections throughout life. Include examples of how learning, injury recovery, and experiences shape brain structure.",

    "Describe the main principles of object-oriented programming including encapsulation, inheritance, polymorphism, and abstraction. Provide examples of how these principles improve code organization and maintainability.",

    "What are the key factors that contribute to climate change? Discuss greenhouse gas emissions, deforestation, ocean acidification, and feedback loops, and explain the scientific consensus on anthropogenic climate change."
]


def benchmark_prompt(prompt, aggressiveness=0.5):
    """Benchmark a single prompt with compression."""

    # Compress the prompt
    compressed_response = token_client.compress_input(
        input=prompt,
        aggressiveness=aggressiveness,
    )
    compressed_prompt = compressed_response.output

    # Test ORIGINAL prompt
    start_time = time.time()
    messages_original = [HumanMessage(content=prompt)]
    response_original = llm.invoke(messages_original)
    original_time = time.time() - start_time

    # Test COMPRESSED prompt
    start_time = time.time()
    messages_compressed = [HumanMessage(content=compressed_prompt)]
    response_compressed = llm.invoke(messages_compressed)
    compressed_time = time.time() - start_time

    return {
        'original_prompt': prompt,
        'compressed_prompt': compressed_prompt,
        'original_response': response_original.content,
        'compressed_response': response_compressed.content,
        'original_time': original_time,
        'compressed_time': compressed_time,
        'original_chars': len(prompt),
        'compressed_chars': len(compressed_prompt),
        'compression_ratio': len(compressed_prompt) / len(prompt),
    }


def generate_report(results):
    """Generate markdown report with graphs."""

    # Create graphs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Response time comparison
    prompts_idx = list(range(1, len(results) + 1))
    original_times = [r['original_time'] for r in results]
    compressed_times = [r['compressed_time'] for r in results]

    x = range(len(prompts_idx))
    width = 0.35

    ax1.bar([i - width/2 for i in x], original_times, width, label='Original', color='#ff6b6b')
    ax1.bar([i + width/2 for i in x], compressed_times, width, label='Compressed', color='#4ecdc4')
    ax1.set_xlabel('Prompt Number')
    ax1.set_ylabel('Response Time (seconds)')
    ax1.set_title('Gemini Response Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(prompts_idx)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Token/character count comparison
    original_chars = [r['original_chars'] for r in results]
    compressed_chars = [r['compressed_chars'] for r in results]

    ax2.bar([i - width/2 for i in x], original_chars, width, label='Original', color='#ff6b6b')
    ax2.bar([i + width/2 for i in x], compressed_chars, width, label='Compressed', color='#4ecdc4')
    ax2.set_xlabel('Prompt Number')
    ax2.set_ylabel('Character Count')
    ax2.set_title('Prompt Size Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(prompts_idx)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('compression_benchmark.png', dpi=300, bbox_inches='tight')
    print("✓ Saved graphs to compression_benchmark.png")

    # Generate markdown report
    avg_compression = sum(r['compression_ratio'] for r in results) / len(results)
    avg_time_saved = sum(r['original_time'] - r['compressed_time'] for r in results) / len(results)
    total_chars_saved = sum(r['original_chars'] - r['compressed_chars'] for r in results)

    markdown = f"""# Bear-1 Compression Benchmark Report

## Summary Statistics

| Metric | Value |
|--------|-------|
| **Prompts Tested** | {len(results)} |
| **Avg Compression Ratio** | {avg_compression:.1%} |
| **Avg Time Saved** | {avg_time_saved:.2f}s |
| **Total Characters Saved** | {total_chars_saved:,} |

![Benchmark Results](compression_benchmark.png)

---

## Detailed Results

"""

    for i, result in enumerate(results, 1):
        time_diff = result['original_time'] - result['compressed_time']
        time_change = (time_diff / result['original_time']) * 100 if result['original_time'] > 0 else 0

        markdown += f"""### Prompt {i}

**Original:** {result['original_prompt'][:100]}...

**Compressed:** {result['compressed_prompt'][:100]}...

| Metric | Original | Compressed | Change |
|--------|----------|------------|--------|
| Characters | {result['original_chars']} | {result['compressed_chars']} | {result['compression_ratio']:.1%} |
| Response Time | {result['original_time']:.2f}s | {result['compressed_time']:.2f}s | {time_change:+.1f}% |

<details>
<summary>View Full Responses</summary>

**Original Response:**
```
{result['original_response'][:300]}...
```

**Compressed Response:**
```
{result['compressed_response'][:300]}...
```

</details>

---

"""

    with open('COMPRESSION_REPORT.md', 'w') as f:
        f.write(markdown)

    print("✓ Saved report to COMPRESSION_REPORT.md")


if __name__ == "__main__":
    print("=" * 80)
    print("BEAR-1 COMPRESSION BENCHMARK")
    print("=" * 80)
    print(f"\nTesting {len(TEST_PROMPTS)} prompts...\n")

    results = []

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"[{i}/{len(TEST_PROMPTS)}] Testing prompt {i}...")
        result = benchmark_prompt(prompt)
        results.append(result)
        print(f"    ✓ Original: {result['original_time']:.2f}s ({result['original_chars']} chars)")
        print(f"    ✓ Compressed: {result['compressed_time']:.2f}s ({result['compressed_chars']} chars)")
        print(f"    ✓ Compression: {result['compression_ratio']:.1%}\n")

    print("\nGenerating report...")
    generate_report(results)
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE!")
    print("=" * 80)
