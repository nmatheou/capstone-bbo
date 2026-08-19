import ast
import re

with open("C:/Users/NtecD/OneDrive/AI/Imperial/Capstone/antigravity/results_week2/inputs.txt", "r") as f:
    content = f.read()

content = content.replace("array(", "").replace(")", "")
# Find closing bracket followed by whitespace/newlines and opening bracket
content = re.sub(r'\]\s*\n\s*\[', '],\n[', content)
try:
    all_lists = ast.literal_eval(f"[{content}]")
    print(f"Parsed {len(all_lists)} lists")
    print("Latest list:", all_lists[-1])
except Exception as e:
    print("Error:", e)
