import re

# T14
f = 'atlas/multiobjective/visualization.py'
with open(f, 'r', encoding='utf8') as fp: content = fp.read()
content = content.replace('import matplotlib.pyplot as plt', 'import matplotlib\nmatplotlib.rcParams["mathtext.default"] = "regular"  # avoid bold math\nimport matplotlib.pyplot as plt')
content = content.replace('("f₁", "f₂")', '("$f_1$", "$f_2$")')
content = content.replace('("f₁", "f₂", "f₃")', '("$f_1$", "$f_2$", "$f_3$")')
with open(f, 'w', encoding='utf8') as fp: fp.write(content)

# T16
f = 'README.md'
with open(f, 'r', encoding='utf8') as fp: content = fp.read()
content = content.replace('**24 built-in algorithms**: 12 iteration-based variants and 12 NFE-based variants', '**18 built-in algorithms**: 18 modern metaheuristic algorithms, all supporting both iteration-based and NFE-based stopping criteria')
content = re.sub(r'1\. Add the iteration version under `atlas/algorithms/iteration/my_algo\.py`\.\n2\. Add the NFE version under `atlas/algorithms/nfe/my_algo\.py`\.\n3\. Register with `@register_algorithm\("my_algo"\)` and `@register_algorithm\("my_algo_nfe"\)`\.\n4\. Import the classes in:\n   - `atlas/algorithms/iteration/__init__\.py`\n   - `atlas/algorithms/nfe/__init__\.py`\n   - `atlas/algorithms/__init__\.py`\n5\. Add tests in `tests/test_algorithms\.py` and `tests/test_nfe_algorithms\.py`\.',
'''1. Add the algorithm file under `atlas/algorithms/iteration/my_algo.py`. Since ATLAS 0.2, only ONE file in `iteration/` is needed — the `BaseAlgorithm` class handles both stopping modes automatically.
2. Register with `@register_algorithm("my_algo", aliases=["my_algo_nfe"])` to preserve backward compatibility.
3. Import the classes in:
   - `atlas/algorithms/iteration/__init__.py`
   - `atlas/algorithms/__init__.py`
4. Add tests in `tests/test_algorithms_all.py`.''', content)
with open(f, 'w', encoding='utf8') as fp: fp.write(content)
