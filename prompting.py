"""Prompt-format contracts for rollout generation.

Hugging Face tokenizers may ship a chat template even when the corresponding
checkpoint is a Base model.  Template presence is therefore not evidence that
the weights learned that protocol: callers must select ``text`` or ``chat``.
"""

from collections.abc import Mapping


PROMPT_FORMATS = ("text", "chat")
PROMPT_FORMAT_CHOICES = ("auto", *PROMPT_FORMATS)
DEFAULT_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}


def resolve_prompt_format(prompt_format, model_source):
    """Resolve the convenience ``auto`` mode from the checkpoint name.

    There is no standardized Hugging Face field recording whether a checkpoint
    was chat-trained.  The ``-Base`` convention covers the Qwen checkpoints we
    use; experiment configs should still be explicit whenever correctness is
    important or the model naming convention is ambiguous.
    """
    if prompt_format not in PROMPT_FORMAT_CHOICES:
        choices = ", ".join(PROMPT_FORMAT_CHOICES)
        raise ValueError(
            f"prompt format must be one of {choices}, not {prompt_format!r}")
    if prompt_format != "auto":
        return prompt_format
    model_name = str(model_source).rstrip("/").rsplit("/", 1)[-1].lower()
    return "text" if model_name.endswith("-base") else "chat"


def render_text_prompt(messages):
    """Turn message-shaped environment data into a plain LM continuation."""
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError("text prompt format requires a non-empty message list")
    contents = []
    for message in messages:
        if (not isinstance(message, Mapping)
                or not isinstance(message.get("content"), str)):
            raise ValueError("text prompt format requires text-only messages")
        contents.append(message["content"])
    text = "\n\n".join(contents)
    return text if text.endswith("\n") else text + "\n"


def _chat_tokens(
        tokenizer, messages, chat_template_kwargs, *,
        add_generation_prompt):
    tokens = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=False,
        **(chat_template_kwargs or {}),
    )
    # Accommodate tokenizers/wrappers that return a BatchEncoding despite
    # return_dict=False; the canonical Transformers return value is a list.
    if isinstance(tokens, Mapping):
        tokens = tokens["input_ids"]
    return list(tokens)


def validate_prompt_contract(
        tokenizer, prompt_format, chat_template_kwargs=None, stop_ids=None):
    """Fail early when the selected protocol cannot terminate correctly.

    ``stop_ids`` are the ids that end a generated turn (the model's list, or
    the tokenizer's eos by default).
    """
    if prompt_format not in PROMPT_FORMATS:
        raise ValueError(
            f"resolved prompt format must be one of {PROMPT_FORMATS}, "
            f"not {prompt_format!r}")
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError(
            f"prompt_format={prompt_format} requires tokenizer.eos_token_id")
    stop_ids = (eos_id,) if stop_ids is None else tuple(stop_ids)
    if prompt_format == "chat":
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError(
                "prompt_format=chat requires a tokenizer chat template")
        # A template closes an assistant turn either with a stop token of
        # its own (Qwen's <|im_end|>) or with the header of the next turn
        # (GLM's <|user|>), so the check renders a completed turn followed
        # by the next user turn and looks for a stop id past the shared
        # prefix with the generation prompt.
        user = {"role": "user", "content": "prompt-contract-user"}
        assistant = {"role": "assistant", "content": "prompt-contract-assistant"}
        generation_prompt = _chat_tokens(
            tokenizer, [user], chat_template_kwargs, add_generation_prompt=True)
        conversation = _chat_tokens(
            tokenizer, [user, assistant, user], chat_template_kwargs,
            add_generation_prompt=False)
        shared = next(
            (index for index, (a, b) in enumerate(zip(generation_prompt, conversation))
             if a != b),
            min(len(generation_prompt), len(conversation)))
        if not set(stop_ids).intersection(conversation[shared:]):
            eos = tokenizer.convert_ids_to_tokens(eos_id)
            stops = list(zip(stop_ids, tokenizer.convert_ids_to_tokens(list(stop_ids))))
            raise ValueError(
                "chat template does not emit tokenizer.eos_token_id="
                f"{eos_id} ({eos!r}) or any other stop id after a completed "
                f"assistant turn (stop ids: {stops}); this usually means a "
                "chat template is being applied to Base weights")


def tokenize_prompt(
        messages, tokenizer, prompt_format, chat_template_kwargs=None):
    """Serialize one generation prompt under an already-resolved format."""
    if prompt_format == "text":
        tokens = list(tokenizer.encode(
            render_text_prompt(messages), add_special_tokens=False))
        special_ids = set(getattr(tokenizer, "all_special_ids", ()))
        injected = sorted(special_ids.intersection(tokens))
        if injected:
            names = tokenizer.convert_ids_to_tokens(injected)
            raise ValueError(
                "text prompt encoded special/control tokens: "
                f"{list(zip(injected, names))}; Base prompts must be plain text")
        return tokens
    if prompt_format == "chat":
        return _chat_tokens(
            tokenizer, messages, chat_template_kwargs,
            add_generation_prompt=True)
    raise ValueError(
        f"resolved prompt format must be one of {PROMPT_FORMATS}, "
        f"not {prompt_format!r}")


def prompt_diagnostic(
        tokenizer, prompt_format, prompt_tokens, max_chars=1000, stop_ids=None):
    """Human-readable startup record of the exact generation prefix and the
    stop ids."""
    rendered = tokenizer.decode(prompt_tokens, skip_special_tokens=False)
    if len(rendered) > max_chars:
        rendered = rendered[:max_chars] + "..."
    eos_id = tokenizer.eos_token_id
    eos = tokenizer.convert_ids_to_tokens(eos_id)
    stop_ids = (eos_id,) if stop_ids is None else tuple(stop_ids)
    return (
        f"prompt_format={prompt_format} eos_token={eos!r} eos_id={eos_id} "
        f"stop_ids={stop_ids} prompt={rendered!r}")
