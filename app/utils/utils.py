def str_to_float(value, default=0.0) -> float:
    """
    Преобразует входное значение в число с плавающей точкой (float).
    Если преобразование невозможно (например, если value None или строка не число),
    возвращает заданное значение по умолчанию.

    Args:
        value: Значение для преобразования (обычно строка).
        default: Значение, возвращаемое при ошибке преобразования (по умолчанию 0.0).

    Returns:
        float: Преобразованное значение или default.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def str_to_int(value, default=0) -> int:
    """
    Преобразует входное значение в целое число (int).
    Если преобразование невозможно (например, если value None или строка не число),
    возвращает заданное значение по умолчанию.

    Args:
        value: Значение для преобразования (обычно строка).
        default: Значение, возвращаемое при ошибке преобразования (по умолчанию 0).

    Returns:
        int: Преобразованное значение или default.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
