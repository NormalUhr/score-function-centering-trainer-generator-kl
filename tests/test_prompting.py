"""Prompt protocols are explicit, special-token safe, and EOS-compatible."""

import pytest

import prompting


class FakeTokenizer:
    eos_token_id = 999
    pad_token_id = 0
    all_special_ids = [990, 991, 999]
    chat_template = "fake-template"

    def __init__(self, chat_emits_eos=True):
        self.chat_emits_eos = chat_emits_eos
        self.last_template_kwargs = None

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        if "<control>" in text:
            return [990]
        return list(text.encode())

    def apply_chat_template(
            self, messages, *, add_generation_prompt, tokenize,
            return_dict, **kwargs):
        assert tokenize is True
        assert return_dict is False
        self.last_template_kwargs = kwargs
        tokens = [990, 1, 991]
        if not add_generation_prompt and self.chat_emits_eos:
            tokens.append(self.eos_token_id)
        return tokens

    def convert_ids_to_tokens(self, token_ids):
        names = {990: "<start>", 991: "<turn-end>", 999: "<eos>"}
        if isinstance(token_ids, int):
            return names[token_ids]
        return [names[token_id] for token_id in token_ids]

    def decode(self, token_ids, skip_special_tokens):
        assert skip_special_tokens is False
        return "decoded:" + ",".join(map(str, token_ids))


def test_resolve_prompt_format():
    assert prompting.resolve_prompt_format(
        "auto", "Qwen/Qwen3-30B-A3B-Base") == "text"
    assert prompting.resolve_prompt_format(
        "auto", "Qwen/Qwen3-0.6B") == "chat"
    assert prompting.resolve_prompt_format("text", "ambiguous") == "text"
    with pytest.raises(ValueError, match="prompt format must be one of"):
        prompting.resolve_prompt_format("messages", "model")


def test_text_prompt_is_plain_document_continuation():
    messages = [
        {"role": "system", "content": "Solve carefully."},
        {"role": "user", "content": "What is 2 + 3?"},
    ]
    assert prompting.render_text_prompt(messages) == (
        "Solve carefully.\n\nWhat is 2 + 3?\n")
    tokenizer = FakeTokenizer()
    tokens = prompting.tokenize_prompt(messages, tokenizer, "text")
    assert bytes(tokens).decode() == "Solve carefully.\n\nWhat is 2 + 3?\n"
    assert not set(tokens).intersection(tokenizer.all_special_ids)


def test_text_prompt_rejects_control_tokens():
    tokenizer = FakeTokenizer()
    with pytest.raises(ValueError, match="special/control tokens"):
        prompting.tokenize_prompt(
            [{"role": "user", "content": "bad <control>"}],
            tokenizer,
            "text",
        )


def test_chat_contract_requires_template_eos():
    kwargs = {"enable_thinking": False}
    tokenizer = FakeTokenizer(chat_emits_eos=True)
    prompting.validate_prompt_contract(tokenizer, "chat", kwargs)
    assert tokenizer.last_template_kwargs == kwargs

    incompatible = FakeTokenizer(chat_emits_eos=False)
    with pytest.raises(ValueError, match="chat template does not emit"):
        prompting.validate_prompt_contract(incompatible, "chat", kwargs)


def test_chat_tokenization_and_diagnostic():
    tokenizer = FakeTokenizer()
    tokens = prompting.tokenize_prompt(
        [{"role": "user", "content": "hello"}],
        tokenizer,
        "chat",
        {"enable_thinking": False},
    )
    assert tokens == [990, 1, 991]
    assert "prompt_format=chat" in prompting.prompt_diagnostic(
        tokenizer, "chat", tokens)
