def dict_to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(
            **{key: dict_to_namespace(value) for key, value in obj.items()}
        )

    if isinstance(obj, list):
        return [dict_to_namespace(item) for item in obj]

    return obj


b = dict_to_namespace(a)
b.hello = "how"
