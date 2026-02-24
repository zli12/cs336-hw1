from __future__ import annotations

import json
from collections.abc import Iterable, Iterator

import regex as re


PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 bytes-to-unicode mapping."""
    # GPT-2 stores vocab/merges using printable unicode "proxies" for raw bytes.
    # This helper recreates that bijection so we can serialize/deserialize safely
    # without losing arbitrary byte values.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(codepoint) for codepoint in cs]))


class Tokenizer:
    """Byte-level BPE tokenizer with optional special-token handling.

    High-level flow when encoding:
    1) Split text into "pre-tokens" using the GPT-2 regex pattern.
    2) Convert each pre-token to UTF-8 bytes.
    3) Apply BPE merges (in merge-list order) inside each pre-token.
    4) Map final byte tokens to integer ids.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        """Build a tokenizer from an in-memory vocab + merge list.

        Args:
            vocab: Mapping from token id -> token bytes.
            merges: Ordered list of merges; earlier entries have higher priority.
            special_tokens: Strings that must be preserved as single tokens.
        """
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges = list(merges)

        # Reverse index used during encoding: token bytes -> token id.
        self.token_to_id: dict[bytes, int] = {token: token_id for token_id, token in self.vocab.items()}

        self.special_tokens = special_tokens or []
        self.special_token_to_id: dict[str, int] = {}

        # Ensure special tokens are present in the vocabulary.
        for special_token in self.special_tokens:
            special_token_bytes = special_token.encode("utf-8")
            if special_token_bytes in self.token_to_id:
                token_id = self.token_to_id[special_token_bytes]
            else:
                token_id = len(self.vocab)
                self.vocab[token_id] = special_token_bytes
                self.token_to_id[special_token_bytes] = token_id
            self.special_token_to_id[special_token] = token_id

        # Merge rank lets us quickly ask: "is this adjacent pair mergeable, and how early?"
        # Lower rank means higher merge priority.
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }
        # Cache BPE results for repeated pre-tokens (a common case in natural language).
        self._bpe_cache: dict[bytes, tuple[bytes, ...]] = {}

        if self.special_tokens:
            # Sort by length desc so overlapping specials prefer the longest match first.
            escaped = [re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)]
            self.special_pattern = re.compile("|".join(escaped))
        else:
            self.special_pattern = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """Load GPT-2-style serialized vocab/merges and construct a tokenizer."""
        # Serialized files use GPT-2 byte->unicode remapping; invert it to get raw bytes back.
        byte_decoder = {v: k for k, v in _bytes_to_unicode().items()}

        with open(vocab_filepath, encoding="utf-8") as vocab_f:
            loaded_vocab = json.load(vocab_f)
        # Convert serialized token strings back to raw bytes.
        vocab = {
            token_id: bytes([byte_decoder[ch] for ch in token_text])
            for token_text, token_id in loaded_vocab.items()
        }

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as merges_f:
            for line in merges_f:
                stripped = line.rstrip()
                parts = stripped.split(" ")
                if not stripped or len(parts) != 2:
                    # Skip empty lines or headers/comments if present.
                    continue
                left, right = parts
                # Each merge line stores exactly two tokens; convert both back to raw bytes.
                merges.append(
                    (
                        bytes([byte_decoder[ch] for ch in left]),
                        bytes([byte_decoder[ch] for ch in right]),
                    )
                )

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _apply_bpe(self, pretoken_bytes: bytes) -> tuple[bytes, ...]:
        """Apply iterative BPE merging to one pre-token (in bytes form)."""
        if pretoken_bytes in self._bpe_cache:
            return self._bpe_cache[pretoken_bytes]

        # Start from byte-level symbols (one byte token per element).
        # Example: b"lower" -> [b"l", b"o", b"w", b"e", b"r"].
        #
        # Worked trace (toy ranks):
        #   symbols_0 = [b"l", b"o", b"w", b"e", b"r"]
        #   available pairs = (l,o), (o,w), (w,e), (e,r)
        #   suppose rank(o,w) < rank(l,o) < rank(e,r) and (w,e) not mergeable
        #   merge (o,w) first -> symbols_1 = [b"l", b"ow", b"e", b"r"]
        #   now pairs are (l,ow), (ow,e), (e,r), then pick lowest-rank among these.
        #
        # Key idea: BPE is greedy by merge rank, repeatedly applied until no adjacent
        # pair in the current sequence appears in merge_ranks.
        symbols: list[bytes] = [bytes([b]) for b in pretoken_bytes]
        if len(symbols) < 2:
            result = tuple(symbols)
            self._bpe_cache[pretoken_bytes] = result
            return result

        while True:
            best_rank = None
            best_pair: tuple[bytes, bytes] | None = None
            # Find the highest-priority mergeable adjacent pair in the current symbol sequence.
            # "Highest priority" means smallest rank in the merge list (earliest learned merge).
            for pair in zip(symbols, symbols[1:]):
                rank = self.merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                # No more applicable merges for this pre-token.
                break

            # Merge every non-overlapping occurrence of the chosen pair in a left-to-right pass.
            # This mirrors standard BPE behavior and avoids ambiguous overlapping merges.
            merged: list[bytes] = []
            i = 0
            while i < len(symbols):
                if (
                    i + 1 < len(symbols)
                    and symbols[i] == best_pair[0]
                    and symbols[i + 1] == best_pair[1]
                ):
                    merged.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged
            if len(symbols) < 2:
                # Single symbol cannot produce any further adjacent pairs.
                break

        result = tuple(symbols)
        self._bpe_cache[pretoken_bytes] = result
        return result

    def _encode_ordinary(self, text: str) -> list[int]:
        """Encode text without special-token shortcuts."""
        ids: list[int] = []
        # Regex pretokenization keeps punctuation/whitespace behavior aligned with GPT-2.
        # We apply BPE *within each pre-token*; merges do not cross pre-token boundaries.
        for match in PATTERN.finditer(text):
            pretoken_bytes = match.group(0).encode("utf-8")
            for symbol in self._apply_bpe(pretoken_bytes):
                ids.append(self.token_to_id[symbol])
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode text into token ids, preserving configured special tokens."""
        if not self.special_pattern:
            return self._encode_ordinary(text)

        ids: list[int] = []
        cursor = 0
        # Alternate between ordinary spans and exact special-token matches.
        # Important: specials bypass BPE and are emitted as fixed ids.
        for match in self.special_pattern.finditer(text):
            start, end = match.span()
            if start > cursor:
                ids.extend(self._encode_ordinary(text[cursor:start]))
            ids.append(self.special_token_to_id[match.group(0)])
            cursor = end
        if cursor < len(text):
            ids.extend(self._encode_ordinary(text[cursor:]))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode chunks from an iterable source (e.g., file handle)."""
        # This keeps memory usage low: we never construct one giant input string.
        # Each chunk is encoded independently and token ids are yielded on the fly.
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back into a UTF-8 string.

        Invalid UTF-8 sequences are replaced with the Unicode replacement character.
        """
        decoded_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return decoded_bytes.decode("utf-8", errors="replace")
