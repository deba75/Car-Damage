"""Canonical taxonomy: 6 parts x {none, low, high} severity per part."""
PARTS = ["front", "rear", "door", "corner", "window", "other"]
SEV_NAMES = ["none", "low", "high"]          # class ids 0 / 1 / 2
UNKNOWN = -1                                  # label not observed -> loss ignored

# Order matters: first match wins ("front door" -> door, "rear window" -> window,
# "front fender" -> corner, "rear wheel" -> other). Generic "rear"/"front" fallbacks
# come after specific part nouns so they don't hijack fenders, wheels, or plates.
# CHECK these against the real class names of every dataset you use (print unmapped!).
RULES = [
    ("window", ["windshield", "windscreen", "window", "glass"]),
    ("door",   ["door", "mirror", "handle"]),
    ("corner", ["fender", "quarter", "rocker", "sill", "side panel", "corner", "pillar"]),
    ("other",  ["wheel", "tire", "tyre", "rim", "roof", "plate", "license"]),
    ("rear",   ["rear bumper", "back bumper", "trunk", "tailgate", "boot", "tail", "rear light", "back light", "rear lamp"]),
    ("front",  ["front bumper", "bumper", "hood", "bonnet", "headlight", "headlamp", "head lamp", "grille", "front light", "front lamp"]),
    ("rear",   ["rear"]),
    ("front",  ["front"]),
    ("corner", ["panel"]),
    ("other",  ["other"]),
]
_unmapped = set()

def map_part(name: str) -> str:
    n = name.lower().replace("_", " ").replace("-", " ").strip()
    for part, keys in RULES:
        if any(k in n for k in keys):
            return part
    _unmapped.add(name)
    return "other"

def unmapped_names():
    return sorted(_unmapped)

def reset_unmapped():
    _unmapped.clear()
