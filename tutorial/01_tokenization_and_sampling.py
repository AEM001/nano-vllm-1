"""
Tutorial 01 — Tokenization and Sampling
========================================
Before a model sees any text, it must be turned into numbers.
After the model produces numbers (logits), they must be turned back into text.
This file covers both ends.

Run it:
    /home/albert/learn/l-vllm/.venv/bin/python tutorial/01_tokenization_and_sampling.py
"""

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# PART 1 — What is a token?
# ---------------------------------------------------------------------------
# A tokenizer splits raw text into chunks called tokens. Each token maps to an
# integer id. The model only ever sees those integers, never raw characters.
#
# We will use a tiny hand-rolled vocabulary to make this concrete.

VOCAB = {
    "<pad>": 0, "<eos>": 1,
    "Hello": 2, "world": 3, "!": 4,
    "the": 5, "cat": 6, "sat": 7,
    "on": 8, "mat": 9, ".": 10,
}
ID_TO_TOKEN = {v: k for k, v in VOCAB.items()}

def encode(text: str) -> list[int]:
    return [VOCAB[w] for w in text.split()]

def decode(ids: list[int]) -> str:
    return " ".join(ID_TO_TOKEN[i] for i in ids)

tokens = encode("Hello world !")
print("=== PART 1: Tokenization ===")
print(f"Text  → ids  : {tokens}")
print(f"ids   → text : {decode(tokens)}")

# Real-world tokenizers (BPE, WordPiece) work on sub-word pieces, but the
# principle is exactly the same: text ↔ list[int].


# ---------------------------------------------------------------------------
# PART 2 — What are logits?
# ---------------------------------------------------------------------------
# At each generation step the model outputs one float per vocabulary token.
# These are called *logits*. Higher = more likely. They are un-normalised scores.

vocab_size = len(VOCAB)
torch.manual_seed(42)
logits = torch.randn(vocab_size)   # pretend these came from the model
print("\n=== PART 2: Logits ===")
print(f"Raw logits: {logits.tolist()}")

probs = F.softmax(logits, dim=-1)
print(f"Probabilities (sum={probs.sum():.4f}): {[f'{p:.3f}' for p in probs.tolist()]}")


# ---------------------------------------------------------------------------
# PART 3 — Greedy vs Temperature sampling
# ---------------------------------------------------------------------------
# Greedy: always pick the most likely token. Deterministic, but boring.
# Temperature: divide logits by T before softmax.
#   T < 1 → sharper distribution → more confident / repetitive
#   T > 1 → flatter  distribution → more random / creative
#   T → 0 → equivalent to greedy
#   T → ∞ → uniform random

def sample_greedy(logits: torch.Tensor) -> int:
    return logits.argmax().item()

def sample_temperature(logits: torch.Tensor, temperature: float) -> int:
    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, 1).item()

print("\n=== PART 3: Greedy vs Temperature sampling ===")
print(f"Greedy pick      : {ID_TO_TOKEN[sample_greedy(logits)]} (id={sample_greedy(logits)})")

torch.manual_seed(0)
for T in [0.1, 0.5, 1.0, 2.0]:
    picks = [ID_TO_TOKEN[sample_temperature(logits, T)] for _ in range(5)]
    print(f"  T={T:.1f}  samples: {picks}")


# ---------------------------------------------------------------------------
# PART 4 — Gumbel-max trick (how nano-vllm's Sampler actually works)
# ---------------------------------------------------------------------------
# Instead of: sample from softmax(logits/T)
# Do:         argmax(logits/T - log(-log(uniform(0,1))))
#
# These are mathematically equivalent but the Gumbel form is faster on GPU
# because argmax is more parallelisable than multinomial.
#
# See: nanovllm/layers/sampler.py

def sample_gumbel(logits: torch.Tensor, temperature: float) -> int:
    scaled = logits / temperature
    # Gumbel noise = -log(-log(U)) where U ~ Uniform(0,1)
    gumbel_noise = -torch.empty_like(scaled).exponential_(1).log()
    return (scaled + gumbel_noise).argmax().item()

# The actual nano-vllm code uses the equivalent form:
#   probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax()
# which avoids the log and works directly in probability space.

print("\n=== PART 4: Gumbel-max trick ===")
torch.manual_seed(0)
gumbel_picks = [ID_TO_TOKEN[sample_gumbel(logits, 1.0)] for _ in range(10)]
torch.manual_seed(0)
normal_picks = [ID_TO_TOKEN[sample_temperature(logits, 1.0)] for _ in range(10)]
print(f"Gumbel  picks: {gumbel_picks}")
print(f"Softmax picks: {normal_picks}")
print("(distributions match, individual samples may differ — they are the same in expectation)")


# ---------------------------------------------------------------------------
# PART 5 — Autoregressive generation loop
# ---------------------------------------------------------------------------
# The model generates one token at a time. Each new token is appended to the
# context and fed back in. This is the decode loop.

def toy_lm(token_ids: list[int]) -> torch.Tensor:
    """
    A fake language model that returns random-ish logits influenced by the
    last token id (just so the output looks slightly meaningful).
    """
    torch.manual_seed(token_ids[-1])
    return torch.randn(vocab_size)

def generate(prompt: str, max_tokens: int = 5, temperature: float = 0.8) -> str:
    ids = encode(prompt)
    print(f"  Prompt ids: {ids}")
    for step in range(max_tokens):
        logits = toy_lm(ids)
        next_id = sample_gumbel(logits, temperature)
        ids.append(next_id)
        print(f"  Step {step+1}: generated '{ID_TO_TOKEN[next_id]}' (id={next_id})")
        if next_id == VOCAB["<eos>"]:
            print("  <eos> hit, stopping.")
            break
    return decode(ids)

print("\n=== PART 5: Autoregressive generation loop ===")
result = generate("Hello world !")
print(f"  Final output: '{result}'")


# ---------------------------------------------------------------------------
# EXPERIMENT ZONE — Try changing these and rerun the file
# ---------------------------------------------------------------------------
# 1. Change the temperature in generate() to 0.1 (very deterministic) vs 5.0
#    (very random). Observe how the sampled tokens change.
#
# 2. Add your own words to VOCAB and VOCAB_TO_ID, then encode/decode new sentences.
#
# 3. Change the logits manually (e.g. logits[VOCAB["cat"]] = 10.0) and see how
#    greedy vs sampling behave differently.
#
# 4. In sample_temperature, print the full probability distribution before and
#    after scaling by T. Watch how it sharpens or flattens.

print("\n=== YOUR EXPERIMENT ===")
custom_logits = torch.zeros(vocab_size)
custom_logits[VOCAB["cat"]] = 5.0    # strongly favour "cat"
custom_logits[VOCAB["mat"]] = 3.0
custom_logits[VOCAB["sat"]] = 1.0

print("Biased logits (cat=5, mat=3, sat=1, rest=0):")
for T in [0.5, 1.0, 2.0]:
    counts = {}
    for _ in range(200):
        tok = ID_TO_TOKEN[sample_temperature(custom_logits, T)]
        counts[tok] = counts.get(tok, 0) + 1
    top = sorted(counts.items(), key=lambda x: -x[1])[:4]
    print(f"  T={T}: {top}")
