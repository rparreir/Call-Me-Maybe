*This project has been created as part of the 42 curriculum by rparreir.*

# Call me maybe

## Description

This project turns natural language into structured function calls. Given a request like
`"What is the sum of 2 and 3?"`, it does not answer `5` — it returns the function to call and
its arguments:

```json
{"name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

The available functions are described in an input file, with their parameter names and types.
The program reads a list of prompts and writes one JSON object per prompt.

The output is always valid JSON matching that schema, and the guarantee does not come from the
prompt — it comes from **constrained decoding**: at every step the program reads the model's
logits and allows only the tokens that keep the output valid, so an invalid token can never be
selected.

The model is `Qwen/Qwen3-0.6B`.

## Instructions

### Requirements

- Python 3.10 or newer
- uv

### Setup

```bash
uv sync
```

The model weights (`Qwen/Qwen3-0.6B`, ~1.2 GB) are downloaded from Hugging Face on the first run
and cached locally, so the first execution takes longer than the following ones.

### Running

```bash
uv run python -m src
```

With explicit paths:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

All three arguments are optional and default to the paths above. The output directory is created
if it does not exist, which is why `data/output/` is not part of the repository.

## Resources

### References

- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B) — the model, its chat template
  and its tokenizer
- Karpathy, [*Let's build the GPT Tokenizer*](https://www.youtube.com/watch?v=zduSFxRajkE) —
  byte-level BPE: why numbers are split digit by digit and why a leading space belongs to the token
- Willard & Louf, [*Efficient Guided Generation for Large Language Models*](https://arxiv.org/abs/2307.09702)
  (2023) — guided generation as a state machine over the vocabulary; the paper behind `outlines`

### How AI was used

I used Claude (Anthropic) as a tutor rather than a code generator:

- **Concepts** — logits, byte-level BPE tokenization, JSON escaping and the JSON number grammar,
  explained until I could restate them myself.
- **Debugging** — pointing at *where* a bug was without fixing it, so I had to find and understand
  each one.
- **Measurement** — small scripts that print the model's logits at a given generation step.
- **Test data** — extra function definitions and prompts covering cases the provided examples do
  not (booleans, decimals, negative numbers, function names that are prefixes of one another).
- **This README** — drafted from my notes and measurements, then reviewed and edited by me.

## Algorithm explanation

At each step the model returns logits for the whole vocabulary. Rather than setting invalid tokens
to −∞, the program takes the argmax over the allowed ids only — same result, cheaper:

```python
def apply_mask(logits, allowed_ids):
    return max(allowed_ids, key=lambda i: logits[i])
```

An invalid token can never be selected because it is never a candidate. And whatever the mask
leaves fully determined is appended to the context without calling the model at all.

The allowed set depends on the declared type, normalised to `number`, `string` or `boolean`:

- `number` — a state machine over the JSON number grammar: sign only first, one dot and only after
  a digit, `,`/`}` only once a digit exists. `float()` never receives `-` or `3.` alone.
- `string` — tokens without a quote are content; a token *containing* one ends the string, keeping
  the text before it; `\"` counts as content.
- `boolean` — candidate filtering over `true` and `false`.

The prompt (`src/build_prompt.py`) uses the model's chat template. It improves *which* function and
values get chosen, but creates no guarantee: an empty prompt still produces valid output.

## Design decisions

**Two separate tracks.** What the model reads and what gets written are different things: the
context holds the prompt, the skeleton and the generated tokens; the result is a Python dict
serialised at the end with `json.dump`, which owns all escaping. The context never reaches the file.

**Termination is a token, not a counter.** The closing quote is part of the allowed set, so the
model decides when a name or a string ends. This is what makes a function name that is a prefix of
another (`fn_get`, `fn_get_square_root`) work: both continuing and stopping are offered.

**Types are normalised at the boundary.** `integer`, `int`, `float` become `number`, `bool` becomes
`boolean`, and anything unknown becomes `string`. Done once per parameter, so the skeleton, the
generator and the closing quote can never disagree about the type.

**Hard caps everywhere.** Constraining the output removes the model's normal way to stop, so every
generation loop has a bounded number of turns. They are a safety net, not a stopping condition.

## Performance analysis

Measured on the provided input files (5 functions, 11 prompts), on a CPU-only laptop under WSL:

| | |
|---|---|
| Correct function selected | 11/11 |
| Correct arguments | 11/11 |
| Structurally valid output | 11/11 — validated against the rules in the subject |
| Total run time | under 3 minutes (limit: 5) |
| Model calls per prompt | ~6 |

**Where the time goes.** Loading the model takes ~6 s and building the id→text map ~1 s; everything
else is forward passes. Each pass costs roughly 2 s on this machine and gets more expensive as the
context grows, so the only real lever is generating fewer tokens. Skipping the passes the mask makes
unnecessary removed 19 of the 24 steps needed to write the five function names, which cut ~49 s from
a full run.

**Reliability.** Valid JSON is structural, not statistical: the mask cannot emit a token that breaks
the schema, so the 100% figure holds for any input, including prompts that make no sense. What can
still be wrong is the *meaning* — which function, which value — and that is bounded by a 0.6B model.

## Challenges faced

**Escaping.** The model writes JSON escapes correctly, so storing the decoded text as-is and
serialising it again escapes everything twice. The text has to be converted into a value before it
is stored.

**Telling my bug from the model's limit.** When a value came out wrong, the cause was usually my own
mask forbidding the token the model wanted — not the model being too small. Printing the top logits
at the failing step is what separates the two cases, and it found every real bug in this project.

**Masking can make things worse.** Removing a token does not make the model pick the one I had in
mind; it picks its next best, which may be further from the answer. Sometimes normalising the result
is better than restricting the choice.

**Stopping.** Constraining the output removes the model's normal way to end a value, so termination
has to be designed as a token it is allowed to choose. Where I used a counter instead, loops ran to
their cap and produced truncated values.

**Time.** Every generated token costs a full forward pass, so the only real lever is generating
fewer tokens — not faster code.

## Testing strategy

- **Baseline.** The provided files are re-run after every change and all 11 results compared with
  the previous run — most bugs showed up as a regression somewhere I had not touched.
- **A validator, not eyeballing.** A script checks the output against the subject's rules: one
  object per prompt in the input order, the three required keys, function names that exist, and
  argument names and types matching the schema.
- **Extra function sets** for what the examples do not cover: booleans, decimals, negative numbers,
  and function names where one is a prefix of another — the last one was a crash the provided files
  could never have revealed.
- **Degenerate inputs.** An empty prompt and a nonsense prompt still produce valid output with a
  real function name, which is the proof that the guarantee is in the mask and not in the prompt.
- **Logit probes.** At a failing step, printing the model's top candidates and whether each is
  allowed says immediately whether the mask or the model is at fault.

## Example usage

An excerpt of `data/input/functions_definition.json`:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
    "returns": {"type": "number"}
  }
]
```

An excerpt of `data/input/function_calling_tests.json`:

```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS"}
]
```

Run:

```bash
uv run python -m src
```

The resulting `data/output/function_calling_results.json`:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS",
    "name": "fn_substitute_string_with_regex",
    "parameters": {
      "source_string": "Hello 34 I'm 233 years old",
      "regex": "([0-9]+)",
      "replacement": "NUMBERS"
    }
  }
]
```

With your own files:

```bash
uv run python -m src \
    --functions_definition my_functions.json \
    --input my_prompts.json \
    --output results.json
```
