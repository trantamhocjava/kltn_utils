import re
from pathlib import Path

path = Path("setup.py")
text = path.read_text(encoding="utf-8")

pattern = r'version\s*=\s*"0\.1\.(\d+)"'

match = re.search(pattern, text)
if not match:
    raise ValueError("Không tìm thấy version dạng 0.1.A trong setup.py")

old_patch = int(match.group(1))
new_patch = old_patch + 1

new_text = re.sub(
    pattern,
    f'version="0.1.{new_patch}"',
    text,
    count=1,
)

path.write_text(new_text, encoding="utf-8")

print(f"Updated version: 0.1.{old_patch} -> 0.1.{new_patch}")
