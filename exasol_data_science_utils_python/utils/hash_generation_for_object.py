from typing import Hashable, Dict, Any, Iterable, Set


def generate_hash_for_object(obj):
    return hash(tuple(_hash_object(v, set())
                      for k, v in sorted(obj.__dict__.items())))


def _hash_object(obj: Any, already_seen: Set[int]) -> int:
    object_id = id(obj)
    if object_id in already_seen:
        print(object_id)
        return 0
    else:
        already_seen.add(object_id)
        if isinstance(obj, Hashable):
            return hash(obj)
        if isinstance(obj, str):
            return hash(obj)
        elif isinstance(obj, Dict):
            return \
                hash(
                    (
                        _hash_object(obj.keys(), already_seen),
                        _hash_object(obj.values(), already_seen)
                    )
                )
        elif isinstance(obj, Iterable):
            return hash(tuple(_hash_object(item, already_seen) for item in obj))
        else:
            return 0
