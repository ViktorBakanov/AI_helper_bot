from difflib import SequenceMatcher

def match_score(text: str, query: str) -> float:
    """
    Сравнивает текст и запрос, возвращая коэффициент похожести от 0 до 1.
    """
    return SequenceMatcher(None, text.lower(), query.lower()).ratio()