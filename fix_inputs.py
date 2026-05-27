import re

with open('app.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'number_input' in line:
        # Skip lines that are already clean (no min/max positional args)
        # Skip ball mills lines — they use min_value/max_value kwargs already
        if 'ball_mills' in line or 'min_value=' in line or 'max_value=' in line:
            new_lines.append(line)
            continue
        # Match: .number_input("label", num, num, num) or with step=
        m = re.search(r'(number_input\()(".*?")\s*,\s*[-\d.]+\s*,\s*[-\d.]+\s*,\s*([-\d.]+)(.*?)(\))', line)
        if m:
            prefix_in_line = line[:line.index('number_input(')]
            rest = line[line.index('number_input('):]
            # get label, default, and optional step
            label = m.group(2)
            default = m.group(3)
            extra = m.group(4)  # could contain step=...
            step_match = re.search(r'step=([\d.]+)', extra)
            step_part = f', step={step_match.group(1)}' if step_match else ''
            new_call = f'number_input({label}, value={default}{step_part})'
            # preserve the variable assignment before number_input
            line = prefix_in_line + new_call + '\n'
    new_lines.append(line)

with open('app.py', 'w') as f:
    f.writelines(new_lines)

print('Done! All min/max limits removed.')
