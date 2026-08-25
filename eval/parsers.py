import re

ANSWER_PATTERNS = [
    re.compile(r"\\boxed\{([^{}]+)\}"),
    re.compile(r"####\s*([^\n]+)"),
    re.compile(r"答案[:：]\s*([^\n]+)"),
]


def _clean(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace("$", "")
    return cleaned.strip()


class Parser:
    @staticmethod
    def extract_answer_gsm8k(text: str) -> str:
        for pattern in ANSWER_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                candidate = matches[-1]
                cleaned = _clean(candidate)
                if cleaned:
                    return cleaned
        numeric_matches = re.findall(r"-?\d+(?:\.\d+)?", text)
        if numeric_matches:
            return _clean(numeric_matches[-1])
        return text.strip()


def is_equiv(str1: str, str2: str, verbose: bool = False) -> bool:
    norm1 = _clean(str1)
    norm2 = _clean(str2)
    if not norm1 or not norm2:
        return norm1 == norm2
    try:
        return abs(float(norm1) - float(norm2)) <= 1e-6
    except ValueError:
        if verbose:
            print(f"Non numeric comparison: '{norm1}' vs '{norm2}'")
        return norm1 == norm2
