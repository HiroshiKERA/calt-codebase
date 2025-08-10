# Tokenizer Guide

This page documents the tokenizer used in **calt** and how to register a **custom vocabulary** (custom tokens).  
The implementation is centered on `calt.data_loader.utils.tokenizer.set_tokenizer`, which returns a `PreTrainedTokenizerFast`.

---

## 1. Goals & Assumptions

- The library targets **polynomials and integer arithmetic** for Transformer models. We use a **simple WordLevel** token stream as the internal representation.
- Tokens are concatenated with **exactly one half-width space** between them.
- **Schema note (customizable):** By default, the *preprocessor* emits tokens using the convention  
  **coefficients → `C<int>`**, **exponents → `E<int>`** (e.g., `C3`, `E2`).  
  This **is a preprocessor convention, not a tokenizer constraint**. You can replace it with your own schema as long as the resulting strings exist in the tokenizer vocabulary and follow the one‑space delimiter rule.
- Typical default internal tokens:
  - Coefficients: `C<integer>` (e.g., `C3`, `C-1`)
  - Exponents: `E<nonnegative integer>` (e.g., `E0`, `E2`)
  - Intra-sequence separator: `[SEP]` (a *regular* token unless you explicitly mark it as special)
- The tokenizer automatically adds **BOS/EOS** at the boundaries (`<s> ... </s>`).

> Example) Polynomial \(3x^2 - 3x\) in the *default* internal form  
> `C3 E2 C-3 E1`  
> → Encoded (with specials): `<s> C3 E2 C-3 E1 </s>`

---

## 2. Quickstart

```python
from calt.data_loader.utils.tokenizer import set_tokenizer

# Example 1: integers ZZ, coefficients within ±50, max degree 5, max sequence length 512
tok_zz = set_tokenizer(field="ZZ", max_coeff=50, max_degree=5, max_length=512)

text = "C3 E2 C-3 E1"
ids = tok_zz.encode(text)
print(tok_zz.convert_ids_to_tokens(ids))  # ['<s>', 'C3', 'E2', 'C-3', 'E1', '</s>']

# Example 2: finite field GF(31), max degree 10
tok_gf = set_tokenizer(field="GF31", max_degree=10, max_length=512)
```

> By design, **padding is enabled** and **truncation is disabled** by default (the collator typically handles dynamic padding/truncation).

---

## 3. Default Tokenization

### 3.1 Model & Preprocessing
- **Model**: `tokenizers.models.WordLevel` (fixed vocabulary)
- **Pre-tokenizer**: `CharDelimiterSplit(" ")` (**split on a single space**)
- **Post-processor**: `TemplateProcessing`
  - Single sequence: `"<s> $A </s>"` (automatically prepends BOS and appends EOS)

### 3.2 Special tokens

The default special tokens are defined via `special_vocab` and registered on the tokenizer.

| Name | String | Placement & Behavior | Notes |
|---|---|---|---|
| PAD | `[PAD]` | Used for **padding** to a uniform length. Typically masked out in attention and loss. | Set as `pad_token`. |
| BOS | `<s>` | **Beginning-of-sequence** token; **automatically prepended** by the post-processor. Do **not** write it in your raw text. | Set as `bos_token`. |
| EOS | `</s>` | **End-of-sequence** token; **automatically appended** by the post-processor. Do **not** write it in your raw text. | Set as `eos_token`. |
| CLS | `[CLS]` | Optional **classification** token. Included for compatibility with encoder-style pipelines. Not used by default in calt. | Set as `cls_token`. |

> **Important:** `[SEP]` is **not** a special token by default. It is a *regular* token included in the vocabulary to delimit algebraic segments (e.g., separating factors). If you want `[SEP]` to behave like a special token, add it via `add_special_tokens` under `additional_special_tokens`.

### 3.3 Default vocabulary generation

When `vocab_config` is **not** provided, `set_tokenizer` generates a vocabulary from the following rules:

- Coefficient tokens `CONSTS` (default schema)
  - `field="ZZ"` (integers): `C-<max_coeff> ... C0 ... C<max_coeff>`
  - `field="GF<p>"` (finite field): `C-(p-1) ... C-1 C0 C1 ... C(p-1)`  
    (Arithmetic is interpreted modulo `p`. The negative forms appear for symmetry in the vocabulary.)
- Exponent tokens `ECONSTS` (default schema)
  - `E0, E1, ..., E<max_degree>`
- Final vocabulary layout
  - `["[C]"] + CONSTS + ECONSTS + ["[SEP]"] + (special tokens)`
  - `"[C]"` is a *regular* sentinel token reserved by some generators; you can ignore it if unused.

> **WordLevel caution:** tokens **must** exist in the vocabulary. Any out-of-vocabulary string **cannot** be encoded.

### 3.4 Customizing the internal token schema (beyond `C*`/`E*`)

The **token schema is defined by the preprocessor**, not by the tokenizer. To use a different schema (e.g., `K<int>` for coefficients and `P<int>` for exponents):

1. **Make your preprocessor/generator emit** the desired strings, separated by single spaces.  
   Example text: `K3 P2 K-3 P1`
2. **Provide a matching vocabulary**, preferably via YAML (see §4.1), that includes all tokens your data can produce (e.g., `"K-50"... "K50"`, `"P0"... "P10"`, plus any separators like `"[SEP]"`).  
3. The tokenizer will then treat these tokens as regular WordLevel entries. BOS/EOS handling remains unchanged.

---

## 4. Registering a custom vocabulary

There are two common approaches.

### 4.1 Provide the vocabulary via YAML (recommended)

Prepare a minimal YAML file with the **token list** and **special tokens mapping**:

```yaml
vocab:
  - "[C]"
  - "C-50"   # ...
  - "C0"
  - "C50"
  - "E0"
  - "E1"     # ...
  - "E10"
  - "[SEP]"

special_vocab:
  pad_token: "[PAD]"
  bos_token: "<s>"
  eos_token: "</s>"
  cls_token: "[CLS]"
```

Then load it and pass the dict to `set_tokenizer`:

```python
import yaml
from calt.data_loader.utils.tokenizer import set_tokenizer

with open("config/vocab.yaml", "r") as f:
    vocab_config = yaml.safe_load(f)

tok = set_tokenizer(vocab_config=vocab_config, max_length=512)
```

- This method **fully fixes** the vocabulary (no OOV at encode time).
- Useful when you need strict control over admissible coefficients/degrees.
- For a **custom schema** (e.g., `K*`/`P*`), define those tokens in YAML instead of `C*`/`E*`.

### 4.2 Add tokens to an existing tokenizer

Start from the default vocabulary, then add **regular tokens** or **additional special tokens**:

```python
# Add regular tokens
num_added = tok.add_tokens(["C-101", "C101", "E11"])

# Add additional special tokens
tok.add_special_tokens({
    "additional_special_tokens": ["[VAR_X]", "[VAR_Y]"]
})
```

> If you add tokens after model initialization, remember to resize the token embeddings on the model side (e.g., `model.resize_token_embeddings(len(tok))`).

---

## 5. Parameters (API)

```python
set_tokenizer(
    field: str = "GF",        # "ZZ" or "GF<p>" (e.g., "GF31")
    max_coeff: int = 100,     # upper bound for |coefficients| when field == "ZZ"
    max_degree: int = 10,     # maximum exponent value (0..max_degree)
    max_length: int = 512,    # model_max_length for HF tokenizer
    vocab_config: dict | None = None,  # load from YAML into a dict and pass here
) -> transformers.PreTrainedTokenizerFast
```

- If `vocab_config` is provided, the vocabulary is built **only** from the YAML (no auto-generation).
- For `field="GF<p>"`, `<p>` must be an integer; coefficients are listed as `C-(p-1) ... C(p-1)` in the **default schema**.
- Typical error messages on invalid input:
  - `ValueError: unknown field: ...`
  - `ValueError: Invalid field specification for GF(p): ...`

---

## 6. Operational notes (pitfalls)

- **No OOV**: WordLevel cannot encode unknown strings. Ensure your data generator emits **only in-vocab** tokens.
- **One space delimiter**: The pre-tokenizer splits on **a single space**. Always join tokens with exactly one half-width space.
- **Auto BOS/EOS**: Do **not** write `<s>` or `</s>` in raw text; they are injected automatically.
- **Length control**: Truncation is disabled by default. Keep sequences within `max_length` using your collator/dataloader.

---

## 7. FAQ

**Q. Should variable names (x, y, …) appear as tokens?**  
A. The internal form uses only **coefficients `C*`** and **exponents `E*`** by default, but this is **preprocessor-defined**. You can switch to any schema (e.g., `K*`/`P*`) as long as the tokenizer vocabulary matches it. Variable symbols are usually omitted and handled in rendering.
  
**Q. YAML or auto-generation?**  
A. Use YAML for **reproducibility and strict control**. Use auto-generation for quick experimentation within the default schema.

**Q. Any caveats when adding tokens to an existing model?**  
A. After adding tokens to the tokenizer, call `model.resize_token_embeddings(len(tokenizer))`. New embeddings are randomly initialized; you’ll need fine-tuning.

---

## 8. Minimal test

```python
from calt.data_loader.utils.tokenizer import set_tokenizer

tok = set_tokenizer(field="GF31", max_degree=10, max_length=512)
assert tok.pad_token == "[PAD]"
assert tok.bos_token == "<s>"
assert tok.eos_token == "</s>"

text = "C1 E2 C-3 E1"
ids = tok.encode(text)
assert tok.convert_ids_to_tokens(ids) == ["<s>", "C1", "E2", "C-3", "E1", "</s>"]
```

---

## 9. Integration points

- **Dataset generators / preprocessors** must emit strings that contain **only vocabulary tokens**, separated by single spaces. The `C*`/`E*` convention is just the **default**; you can output any schema that matches the tokenizer vocabulary.
- **Collator** is expected to handle dynamic padding/truncation. If sequences exceed `model_max_length`, reconsider your data design or chunking.
