"""Convert Markdown into WhatsApp's formatting dialect.

LLMs answer in Markdown, but WhatsApp only understands:

    *bold*   _italic_   ~strikethrough~   ```monospace```

Sending raw Markdown leaves literal ``**`` and backticks in the chat, so every
outgoing message is normalised here before it is sent (and before its echo
fingerprint is recorded).
"""

import re

_FENCED_CODE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n?(.*?)```", re.S)
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_LINK = re.compile(r"!?\[([^\]]+)\]\((https?://[^)\s]+)\)")
_HEADING = re.compile(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*$")
_BULLET = re.compile(r"(?m)^[ \t]*[-*+][ \t]+")
_ITALIC = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_BOLD_STAR = re.compile(r"\*\*(.+?)\*\*", re.S)
_BOLD_UNDERSCORE = re.compile(r"__(.+?)__", re.S)
_STRIKE = re.compile(r"~~(.+?)~~", re.S)
_RULE = re.compile(r"(?m)^[ \t]*([-*_])[ \t]*(?:\1[ \t]*){2,}$")
_BLANK_LINES = re.compile(r"\n{3,}")


def _inline(text: str) -> str:
    """Format inline Markdown that is not a heading, bullet or code block."""
    text = _LINK.sub(lambda m: f"{m.group(1)} ({m.group(2)})", text)
    text = _INLINE_CODE.sub(lambda m: f"```{m.group(1)}```", text)

    # Markdown italic (*x*) -> WhatsApp italic (_x_). Runs before bold so the
    # bold markers produced below are not converted a second time.
    text = _ITALIC.sub(r"_\1_", text)

    # Markdown bold (**x** and __x__) -> WhatsApp bold (*x*).
    text = _BOLD_STAR.sub(r"*\1*", text)
    text = _BOLD_UNDERSCORE.sub(r"*\1*", text)

    # Markdown strikethrough (~~x~~) -> WhatsApp strikethrough (~x~).
    text = _STRIKE.sub(r"~\1~", text)
    return text


def to_whatsapp(text: str) -> str:
    """Return *text* using WhatsApp markup instead of Markdown."""
    if not text:
        return text

    # Fenced code blocks must survive the other substitutions untouched.
    blocks: list[str] = []

    def _stash_code(match: re.Match) -> str:
        blocks.append(match.group(2).strip("\n"))
        return f"\x00code{len(blocks) - 1}\x00"

    text = _FENCED_CODE.sub(_stash_code, text)

    # Headings become bold lines. Their inline formatting is handled separately
    # so the bold markers are not converted again by the italic pass.
    headings: list[str] = []

    def _stash_heading(match: re.Match) -> str:
        formatted = _inline(match.group(1).strip())
        # Avoid doubling the bold markers when the heading was already bold.
        if (
            len(formatted) > 2
            and formatted.startswith("*")
            and formatted.endswith("*")
            and not formatted.startswith("**")
        ):
            formatted = formatted[1:-1]
        headings.append(formatted)
        return f"\x00head{len(headings) - 1}\x00"

    text = _HEADING.sub(_stash_heading, text)
    text = _BULLET.sub("• ", text)
    text = _RULE.sub("", text)
    text = _inline(text)

    for index, heading in enumerate(headings):
        text = text.replace(f"\x00head{index}\x00", f"*{heading}*")
    for index, code in enumerate(blocks):
        text = text.replace(f"\x00code{index}\x00", f"```\n{code}\n```")

    return _BLANK_LINES.sub("\n\n", text).strip()
