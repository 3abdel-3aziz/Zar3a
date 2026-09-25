import re
from typing import List, Optional

# ---------------------------------------------------------------------------
# English-article regex (used by LegalArticleParser.extract_article_id)
# Covers:
#   "Article 1"          → group(1) = "1"
#   "Article (4)"        → group(1) = "4"
#   "(Article 1)"        → group(1) = "1"
#   "## Article 1 Title" → group(1) = "1"
# ---------------------------------------------------------------------------
_EN_ARTICLE_RE: re.Pattern = re.compile(
    r"\(?\s*Article\s*\(?\s*(\d+)\s*\)?",
    re.IGNORECASE,
)

# Detects any Arabic character (used to skip English strategy on Arabic text)
_HAS_ARABIC_RE: re.Pattern = re.compile(r"[\u0600-\u06FF]")


class AgriculturalLawParser:
    """
    A production-ready parser designed to extract article numbers 
    from Arabic legal text chunks for Financial Crime Compliance (FCC) documents.
    
    It supports:
    1. Exact digits (e.g., 'مادة 45', 'الماده (12) مكرر', '#### ماده (٦) :')
    2. Textual Arabic numbers even inside brackets (e.g., '#### (الماده الاولي) تسري')
    3. Multi-pass prefix stripping (ال/و/ف/ب)
    4. Suffix detection (مكرر/ثانيا/ثالثا)
    Also supports English patterns: 'Article N', 'Article (N)', '(Article N)'
    """

    def __init__(self) -> None:
        """Initializes mapping dictionaries and compiles optimized RegEx patterns."""
        
        # Mapping dictionaries for normalized Arabic words to numbers.
        # Note: Keys are pre-normalized (ة -> ه, [يى] -> ي, [إأآ] -> ا) to match the pipeline.
        self._units: dict[str, int] = {
            "واحد": 1, "واحده": 1, "اول": 1, "اولي": 1,
            "اثنان": 2, "اثنين": 2, "اثنا": 2, "اثني": 2, "ثاني": 2, "ثانيه": 2,
            "ثلاثه": 3, "ثلاث": 3, "ثالع": 3, "ثالث": 3, "ثالم": 3, "ثالثه": 3, # بدائل متوقعة من عيوب الـ OCR
            "اربعه": 4, "اربع": 4, "رابع": 4, "رابعه": 4,
            "خمسه": 5, "خمس": 5, "خامس": 5, "خامسه": 5,
            "سته": 6, "ست": 6, "سادس": 6, "سادسه": 6,
            "سبعه": 7, "سبع": 7, "سابع": 7, "سابعه": 7,
            "ثمانيه": 8, "ثمان": 8, "ثامن": 8, "ثامنه": 8,
            "تسعه": 9, "تسع": 9, "تاسع": 9, "تاسعه": 9,
            "عشره": 10, "عشر": 10, "عاشر": 10, "عاشره": 10
        }
        
        self._tens: dict[str, int] = {
            "عشرون": 20, "عشرين": 20,
            "ثلاثون": 30, "ثلاثين": 30,
            "اربعون": 40, "اربعين": 40,
            "خمسون": 50, "خمسين": 50,
            "ستون": 60, "ستين": 60,
            "سبعون": 70, "سبعين": 70,
            "ثمانون": 80, "ثمانين": 80,
            "تسعون": 90, "تسعين": 90
        }
        
        self._hundreds: dict[str, int] = {
            "مائه": 100, "مئه": 100, "مائتان": 200, "مائتين": 200,
            "ثلاثمائه": 300, "اربعمائه": 400, "خمسمائه": 500,
            "ستمائه": 600, "سبعمائه": 700, "ثمانمائه": 800, "تسعمائه": 900
        }

        # Suffixes that modify the article ID (e.g., Article 5 Bis)
        self._suffixes: dict[str, str] = {
            "مكرر": "bis", "ثانيا": "2", "ثالثا": "3", "رابعا": "4"
        }


        self._numeric_pattern: re.Pattern = re.compile(
                    r"(?:الماد[هة]|ماد[هة])\s*[\(\（]?\s*([0-9\u0660-\u0669]+)", 
                    re.IGNORECASE
                )
        
        # Pattern 2: لقط الأرقام اللفظية (الماده الاولي أو (الماده الثانيه) أو حتى ا ل م ا د ه)
        self._textual_pattern: re.Pattern = re.compile(
            r"(?:الماد[هة]|ماد[هة])\s*[\(\（]?\s*([أ-ي]+)(?:\s*[\)\)]?\s*([أ-ي]+))?", 
            re.IGNORECASE
        )

    def _normalize_token(self, token: str) -> str:
        """
        Strips common Arabic prefixes (ال، و، ف، ب) and normalizes orthography
        to ensure dictionary lookup consistency.
        """
        # 1. Multi-pass prefix stripping (e.g., 'وبالماده' -> 'ماده')
        while len(token) > 2:
            if token.startswith("ال"):
                token = token[2:]
            elif token.startswith("و") or token.startswith("ف") or token.startswith("ب"):
                token = token[1:]
            else:
                break
        
        # 2. Standardize characters (Alef, Ta-Marbuta, Ya)
        token = re.sub(r'[إأآ]', 'ا', token)
        token = re.sub(r'ة\b', 'ه', token)
        token = re.sub(r'[يى]\b', 'ي', token)
        return token

    def _words_to_number(self, text_tokens: List[str]) -> tuple[Optional[int], Optional[str]]:
        """
        Parses Arabic text tokens and computes their mathematical value + suffix.
        Returns: (number, suffix_string)
        """
        total: int = 0
        current: int = 0
        suffix: Optional[str] = None
        
        for raw_token in text_tokens:
            norm_token = self._normalize_token(raw_token)
            
            if norm_token in self._suffixes:
                suffix = self._suffixes[norm_token]
                continue

            if norm_token in self._units:
                current += self._units[norm_token]
            elif norm_token in self._tens:
                current += self._tens[norm_token]
            elif norm_token in self._hundreds:
                total += self._hundreds[norm_token]
            elif norm_token in ["عشر", "عشره"] and 0 < current < 10:
                # Handle 11-19 (e.g., 'حادي عشر')
                current += 10
            elif norm_token == "حادي" or norm_token == "حاديه":
                current += 1
                
        total += current
        return (total if total > 0 else None), suffix

    def _normalize_digits(self, text: str) -> str:
        """Converts Arabic digits (١٢٣) to Western digits (123)."""
        arabic_digits = "٠١٢٣٤٥٦٧٨٩"
        western_digits = "0123456789"
        translation_table = str.maketrans(arabic_digits, western_digits)
        return text.translate(translation_table)

    def extract_article_id(self, chunk_content: str) -> str:
        """
        Scans the beginning of a chunk to detect an article signature.

        Resolution order:
          0. English  – "Article N", "Article (N)", "(Article N)"
          1. Arabic numeric  – مادة (١٢) / مادة 88
          2. Arabic textual  – الماده الاولي
        """
        if not chunk_content:
            return "0"

        sample_text: str = chunk_content[:200].strip()

        # ------------------------------------------------------------------
        # Strategy 0: English article number (only when text is in English)
        # ------------------------------------------------------------------
        if not _HAS_ARABIC_RE.search(sample_text):
            en_match = _EN_ARTICLE_RE.search(sample_text)
            if en_match:
                return str(int(en_match.group(1)))

        # print(f"\n[DEBUG] Input text into parser: {repr(sample_text[:100])}")

        
        # Strategy 1: Check for standard or eastern digits (e.g., ماده (١) أو بماده (٨٨))
        numeric_match = self._numeric_pattern.search(sample_text)
        if numeric_match:
            base_num = self._normalize_digits(numeric_match.group(1))
            
            suffix_str = ""
            for ar_suffix, en_suffix in self._suffixes.items():
                if ar_suffix in sample_text[:100]:
                    suffix_str = f"_{en_suffix}"
                    break
                    
            sub_str = ""
            sub_match = re.search(r"[\(\（]\s*([أ-ي])\s*[\)\）]", sample_text[:100])
            if sub_match:
                sub_letter = sub_match.group(1)
                if sub_letter in ["أ", "ا"]: sub_str = "_a"
                elif sub_letter == "ب": sub_str = "_b"
                elif sub_letter == "ج": sub_str = "_c"
                elif sub_letter == "د": sub_str = "_d"
            
            # print(f"[DEBUG] -> Success Strategy 1: {base_num}{suffix_str}{sub_str}")
            return f"{base_num}{suffix_str}{sub_str}"
            
        # Strategy 2: Check for written Arabic words (e.g., (الماده الاولى) أو الماده الثامنه)
  # Strategy 2: Check for written Arabic words
        textual_match = self._textual_pattern.search(sample_text)
        if textual_match:
            word1 = textual_match.group(1) or ""
            word2 = textual_match.group(2) or ""
            
            raw_matched_text = f"{word1} {word2}".strip()
            # print(f"[DEBUG] -> Found Textual Match raw: {repr(raw_matched_text)}")
            # -------------------------------------
            
            test_tokens = raw_matched_text.split()
            single_char_tokens = [t for t in test_tokens if len(t) == 1]
            
            if len(test_tokens) > 0 and (len(single_char_tokens) / len(test_tokens)) > 0.6:
                condensed = re.sub(r'\s+', '', raw_matched_text)
                for suffix_word in ["مكرر", "ثانيا", "ثالثا", "رابعا"]:
                    if suffix_word in condensed:
                        condensed = condensed.replace(suffix_word, f" {suffix_word}")
                raw_words = condensed.split()
                # print(f"[DEBUG] -> Condensed Spaced Text to: {raw_words}")
            else:
                raw_words = [re.sub(r'[\)\(\]\）\（]', '', w) for w in test_tokens]
            
            clean_tokens: List[str] = [w for w in raw_words if w not in ["بعد", "من", "في"]]
            
            num, suffix_val = self._words_to_number(clean_tokens)
            if num is not None:
                suffix_str = f"_{suffix_val}" if suffix_val else ""
                # print(f"[DEBUG] -> Success Strategy 2: {num}{suffix_str}")
                return f"{num}{suffix_str}"
                
        # print("[DEBUG] -> Failed to catch any article ID. Returning '0'")
        return "0"