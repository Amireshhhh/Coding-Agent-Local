"""Evaluation tasks: small coding tasks, each with a pass/fail check command and a reference solution.

Every task must FAIL its check on the initial repo and PASS after the reference ``solution`` ops are
applied; ``python -m evals.run --self-check`` verifies both properties for all tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CHECK = "python -m pytest -q -p no:cacheprovider"

MUTANT_CHECK = '''import shutil, subprocess, sys, tempfile, pathlib
ok = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"]).returncode == 0
out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "tests"],
                     capture_output=True, text=True).stdout
n = sum(1 for line in out.splitlines() if "::" in line)
if not ok or n < {min_tests}:
    print(f"tests must pass and be >= {min_tests} (passed={{ok}}, collected={{n}})"); sys.exit(1)
tmp = pathlib.Path(tempfile.mkdtemp())
shutil.copytree(".", tmp / "r")
(tmp / "r" / "{target}").write_text(pathlib.Path("mutants/{mutant}").read_text())
killed = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
                        cwd=tmp / "r").returncode != 0
print("mutant killed" if killed else "mutant survived: tests are too weak")
sys.exit(0 if killed else 1)
'''


@dataclass
class Task:
    """One evaluation task."""

    id: str
    category: str
    prompt: str
    files: dict[str, str]
    solution: list[dict[str, Any]]
    check: str = CHECK
    extra: dict[str, Any] = field(default_factory=dict)


def edit(path: str, old: str, new: str) -> dict[str, Any]:
    """Reference ``edit_file`` op."""
    return {"op": "edit", "path": path, "old_str": old, "new_str": new}


def write(path: str, content: str) -> dict[str, Any]:
    """Reference ``write_file`` op."""
    return {"op": "write", "path": path, "content": content}


FIX = "The test suite fails. Fix the bug in the source code (do not modify the tests) so that `{check}` passes."

TASKS: list[Task] = [
    Task("fix_add", "fix_bug", FIX.format(check=CHECK),
         {"calc.py": "def add(a, b):\n    return a - b\n",
          "tests/test_calc.py": "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n    assert add(-1, 1) == 0\n"},
         [edit("calc.py", "return a - b", "return a + b")]),
    Task("fix_off_by_one", "fix_bug", FIX.format(check=CHECK),
         {"series.py": "def sum_to(n):\n    \"\"\"Sum of 1..n inclusive.\"\"\"\n    return sum(range(1, n))\n",
          "tests/test_series.py": "from series import sum_to\n\n\ndef test_sum_to():\n    assert sum_to(1) == 1\n    assert sum_to(10) == 55\n"},
         [edit("series.py", "range(1, n)", "range(1, n + 1)")]),
    Task("fix_reverse", "fix_bug", FIX.format(check=CHECK),
         {"text.py": "def reverse(s):\n    return s[::1]\n",
          "tests/test_text.py": "from text import reverse\n\n\ndef test_reverse():\n    assert reverse('abc') == 'cba'\n    assert reverse('') == ''\n"},
         [edit("text.py", "s[::1]", "s[::-1]")]),
    Task("fix_palindrome_case", "fix_bug", FIX.format(check=CHECK),
         {"pal.py": "def is_palindrome(s):\n    t = ''.join(c for c in s if c.isalnum())\n    return t == t[::-1]\n",
          "tests/test_pal.py": "from pal import is_palindrome\n\n\ndef test_pal():\n    assert is_palindrome('Racecar')\n"
                               "    assert is_palindrome('A man, a plan, a canal: Panama')\n    assert not is_palindrome('abc')\n"},
         [edit("pal.py", "c for c in s if c.isalnum()", "c.lower() for c in s if c.isalnum()")]),
    Task("fix_fizzbuzz", "fix_bug", FIX.format(check=CHECK),
         {"fb.py": "def fizzbuzz(n):\n    if n % 3 == 0:\n        return 'Fizz'\n    if n % 5 == 0:\n        return 'Buzz'\n"
                   "    if n % 15 == 0:\n        return 'FizzBuzz'\n    return str(n)\n",
          "tests/test_fb.py": "from fb import fizzbuzz\n\n\ndef test_fb():\n    assert [fizzbuzz(i) for i in (1, 3, 5, 15)] == "
                              "['1', 'Fizz', 'Buzz', 'FizzBuzz']\n"},
         [edit("fb.py", "    if n % 3 == 0:\n        return 'Fizz'\n",
               "    if n % 15 == 0:\n        return 'FizzBuzz'\n    if n % 3 == 0:\n        return 'Fizz'\n")]),
    Task("fix_average_empty", "fix_bug", FIX.format(check=CHECK),
         {"stats.py": "def average(xs):\n    return sum(xs) / len(xs)\n",
          "tests/test_stats.py": "from stats import average\n\n\ndef test_avg():\n    assert average([1, 2, 3]) == 2\n\n\n"
                                 "def test_empty():\n    assert average([]) == 0.0\n"},
         [edit("stats.py", "    return sum(xs) / len(xs)", "    if not xs:\n        return 0.0\n    return sum(xs) / len(xs)")]),
    Task("fix_merge_mutation", "fix_bug", FIX.format(check=CHECK),
         {"dicts.py": "def merge(a, b):\n    \"\"\"Return a NEW dict with b's keys overriding a's.\"\"\"\n    a.update(b)\n    return a\n",
          "tests/test_dicts.py": "from dicts import merge\n\n\ndef test_merge():\n    a = {'x': 1}\n    out = merge(a, {'y': 2})\n"
                                 "    assert out == {'x': 1, 'y': 2}\n    assert a == {'x': 1}\n"},
         [edit("dicts.py", "    a.update(b)\n    return a\n", "    out = dict(a)\n    out.update(b)\n    return out\n")]),
    Task("fix_parse_int", "fix_bug", FIX.format(check=CHECK),
         {"parse.py": "def parse_int(s, default=None):\n    try:\n        return int(s.replace(' ', ''))\n    except ValueError:\n"
                      "        return default\n",
          "tests/test_parse.py": "from parse import parse_int\n\n\ndef test_parse():\n    assert parse_int(' 42\\n') == 42\n"
                                 "    assert parse_int('1 2') is None\n    assert parse_int('x', 0) == 0\n"},
         [edit("parse.py", "int(s.replace(' ', ''))", "int(s.strip())")]),
    Task("fix_argmax", "fix_bug", FIX.format(check=CHECK),
         {"arr.py": "def argmax(xs):\n    \"\"\"Index of the largest element.\"\"\"\n    return max(xs)\n",
          "tests/test_arr.py": "from arr import argmax\n\n\ndef test_argmax():\n    assert argmax([3, 9, 2]) == 1\n    assert argmax([5]) == 0\n"},
         [edit("arr.py", "return max(xs)", "return max(range(len(xs)), key=lambda i: xs[i])")]),
    Task("fix_count_words", "fix_bug", FIX.format(check=CHECK),
         {"words.py": "def count_words(s):\n    return len([w for w in s.split(' ') if w])\n",
          "tests/test_words.py": "from words import count_words\n\n\ndef test_count():\n    assert count_words('a b\\tc\\nd') == 4\n"
                                 "    assert count_words('   ') == 0\n"},
         [edit("words.py", "s.split(' ')", "s.split()")]),
    Task("fix_config_key", "fix_bug", FIX.format(check=CHECK),
         {"conf.py": "import json\n\n\ndef load_port(text):\n    return int(json.loads(text)['Port'])\n",
          "tests/test_conf.py": "from conf import load_port\n\n\ndef test_port():\n    assert load_port('{\"port\": \"8080\"}') == 8080\n"},
         [edit("conf.py", "['Port']", "['port']")]),
    Task("fix_sort_key", "fix_bug", FIX.format(check=CHECK),
         {"people.py": "def order(people):\n    \"\"\"Sort by age, then name.\"\"\"\n    return sorted(people, key=lambda p: (p['name'], p['age']))\n",
          "tests/test_people.py": "from people import order\n\n\ndef test_order():\n    ps = [{'name': 'b', 'age': 3}, {'name': 'a', 'age': 3}, "
                                  "{'name': 'z', 'age': 1}]\n    assert [p['name'] for p in order(ps)] == ['z', 'a', 'b']\n"},
         [edit("people.py", "(p['name'], p['age'])", "(p['age'], p['name'])")]),
    Task("add_factorial", "add_function",
         f"Add a function factorial(n) to mathutils.py (n >= 0, factorial(0) == 1; raise ValueError for n < 0) so that "
         f"`{CHECK}` passes.",
         {"mathutils.py": "\"\"\"Math helpers.\"\"\"\n",
          "tests/test_mathutils.py": "import pytest\nfrom mathutils import factorial\n\n\ndef test_fact():\n    assert factorial(0) == 1\n"
                                     "    assert factorial(5) == 120\n    with pytest.raises(ValueError):\n        factorial(-1)\n"},
         [write("mathutils.py", "\"\"\"Math helpers.\"\"\"\n\n\ndef factorial(n):\n    if n < 0:\n        raise ValueError('n must be >= 0')\n"
                                "    out = 1\n    for i in range(2, n + 1):\n        out *= i\n    return out\n")]),
    Task("add_slugify", "add_function",
         f"Add slugify(text) to text_utils.py: lowercase, non-alphanumeric runs become a single '-', no leading/trailing '-'. "
         f"Make `{CHECK}` pass.",
         {"text_utils.py": "import re\n",
          "tests/test_slug.py": "from text_utils import slugify\n\n\ndef test_slug():\n    assert slugify('Hello, World!') == 'hello-world'\n"
                                "    assert slugify('  a--b  ') == 'a-b'\n"},
         [write("text_utils.py", "import re\n\n\ndef slugify(text):\n    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')\n")]),
    Task("add_chunk", "add_function",
         f"Add chunk(items, size) to lists.py returning consecutive lists of at most `size` items. Make `{CHECK}` pass.",
         {"lists.py": "",
          "tests/test_lists.py": "from lists import chunk\n\n\ndef test_chunk():\n    assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\n"
                                 "    assert chunk([], 3) == []\n"},
         [write("lists.py", "def chunk(items, size):\n    return [list(items[i:i + size]) for i in range(0, len(items), size)]\n")]),
    Task("add_stack_peek", "add_function",
         f"Add a peek() method to Stack in stack.py that returns the top item without removing it and raises IndexError "
         f"when empty. Make `{CHECK}` pass.",
         {"stack.py": "class Stack:\n    def __init__(self):\n        self._items = []\n\n    def push(self, x):\n        self._items.append(x)\n\n"
                      "    def pop(self):\n        return self._items.pop()\n",
          "tests/test_stack.py": "import pytest\nfrom stack import Stack\n\n\ndef test_peek():\n    s = Stack()\n    s.push(1)\n    s.push(2)\n"
                                 "    assert s.peek() == 2\n    assert s.pop() == 2\n    s.pop()\n    with pytest.raises(IndexError):\n        s.peek()\n"},
         [edit("stack.py", "    def pop(self):\n        return self._items.pop()\n",
               "    def pop(self):\n        return self._items.pop()\n\n    def peek(self):\n        if not self._items:\n"
               "            raise IndexError('peek from empty stack')\n        return self._items[-1]\n")]),
    Task("add_temperature", "add_function",
         f"Add c_to_f(c) and f_to_c(f) to temps.py. Make `{CHECK}` pass.",
         {"temps.py": "",
          "tests/test_temps.py": "from temps import c_to_f, f_to_c\n\n\ndef test_temps():\n    assert c_to_f(100) == 212\n    assert f_to_c(32) == 0\n"
                                 "    assert round(f_to_c(c_to_f(37.5)), 6) == 37.5\n"},
         [write("temps.py", "def c_to_f(c):\n    return c * 9 / 5 + 32\n\n\ndef f_to_c(f):\n    return (f - 32) * 5 / 9\n")]),
    Task("add_is_prime", "add_function",
         f"Add is_prime(n) to primes.py. Make `{CHECK}` pass.",
         {"primes.py": "",
          "tests/test_primes.py": "from primes import is_prime\n\n\ndef test_primes():\n    assert [n for n in range(20) if is_prime(n)] == "
                                  "[2, 3, 5, 7, 11, 13, 17, 19]\n"},
         [write("primes.py", "def is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n    while i * i <= n:\n        if n % i == 0:\n"
                             "            return False\n        i += 1\n    return True\n")]),
    Task("add_flatten", "add_function",
         f"Add flatten(nested) to nest.py that flattens arbitrarily nested lists. Make `{CHECK}` pass.",
         {"nest.py": "",
          "tests/test_nest.py": "from nest import flatten\n\n\ndef test_flatten():\n    assert flatten([1, [2, [3, [4]]], 5]) == [1, 2, 3, 4, 5]\n"
                                "    assert flatten([]) == []\n"},
         [write("nest.py", "def flatten(nested):\n    out = []\n    for x in nested:\n        if isinstance(x, list):\n"
                           "            out.extend(flatten(x))\n        else:\n            out.append(x)\n    return out\n")]),
    Task("refactor_rename", "refactor",
         f"Rename the function calc_total to compute_total everywhere (definition and all callers). Make `{CHECK}` pass.",
         {"billing.py": "def calc_total(items):\n    return sum(i['price'] * i['qty'] for i in items)\n",
          "report.py": "from billing import calc_total\n\n\ndef report(items):\n    return f'Total: {calc_total(items)}'\n",
          "tests/test_billing.py": "import billing\nfrom report import report\n\n\ndef test_rename():\n    assert not hasattr(billing, 'calc_total')\n"
                                   "    assert billing.compute_total([{'price': 2, 'qty': 3}]) == 6\n"
                                   "    assert report([{'price': 1, 'qty': 1}]) == 'Total: 1'\n"},
         [edit("billing.py", "def calc_total", "def compute_total"),
          edit("report.py", "calc_total", "compute_total", ) | {"replace_all": True}]),
    Task("refactor_constant", "refactor",
         f"In pricing.py, replace the magic number 0.2 with a module-level constant TAX_RATE and use it. Make `{CHECK}` pass.",
         {"pricing.py": "def with_tax(price):\n    return round(price * (1 + 0.2), 2)\n",
          "tests/test_pricing.py": "import pricing\n\n\ndef test_const():\n    assert pricing.TAX_RATE == 0.2\n    pricing.TAX_RATE = 0.1\n"
                                   "    assert pricing.with_tax(10) == 11.0\n"},
         [write("pricing.py", "TAX_RATE = 0.2\n\n\ndef with_tax(price):\n    return round(price * (1 + TAX_RATE), 2)\n")]),
    Task("refactor_move", "refactor",
         f"Move format_name from utils.py into a new module names.py. utils.py must keep working by importing it from "
         f"names. Make `{CHECK}` pass.",
         {"utils.py": "def format_name(first, last):\n    return f'{last.upper()}, {first}'\n\n\ndef shout(s):\n    return s.upper() + '!'\n",
          "tests/test_move.py": "import inspect\nimport names\nimport utils\n\n\ndef test_move():\n"
                                "    assert names.format_name('ada', 'lovelace') == 'LOVELACE, ada'\n"
                                "    assert utils.format_name is names.format_name\n"
                                "    assert 'def format_name' not in inspect.getsource(utils)\n    assert utils.shout('a') == 'A!'\n"},
         [write("names.py", "def format_name(first, last):\n    return f'{last.upper()}, {first}'\n"),
          write("utils.py", "from names import format_name  # noqa: F401\n\n\ndef shout(s):\n    return s.upper() + '!'\n")]),
    Task("refactor_class", "refactor",
         f"Refactor counter.py: replace the global-variable functions with a class Counter having increment() and a value "
         f"attribute (starting at 0). Make `{CHECK}` pass.",
         {"counter.py": "_value = 0\n\n\ndef increment():\n    global _value\n    _value += 1\n\n\ndef value():\n    return _value\n",
          "tests/test_counter.py": "from counter import Counter\n\n\ndef test_counter():\n    a, b = Counter(), Counter()\n    a.increment()\n"
                                   "    a.increment()\n    b.increment()\n    assert (a.value, b.value) == (2, 1)\n"},
         [write("counter.py", "class Counter:\n    def __init__(self):\n        self.value = 0\n\n    def increment(self):\n"
                              "        self.value += 1\n")]),
    Task("write_tests_shapes", "write_tests",
         "Write pytest tests in tests/test_shapes.py for shapes.py that check the area of a circle and of a square "
         "(at least 2 tests). The check also verifies your tests catch a bug in a mutated implementation.",
         {"shapes.py": "import math\n\n\ndef circle_area(r):\n    return math.pi * r * r\n\n\ndef square_area(s):\n    return s * s\n",
          "mutants/shapes.py": "import math\n\n\ndef circle_area(r):\n    return math.pi * r * 2\n\n\ndef square_area(s):\n    return s * 4\n",
          "check.py": MUTANT_CHECK.format(min_tests=2, target="shapes.py", mutant="shapes.py"),
          "tests/__init__.py": ""},
         [write("tests/test_shapes.py", "import math\nfrom shapes import circle_area, square_area\n\n\ndef test_circle():\n"
                                        "    assert math.isclose(circle_area(3), math.pi * 9)\n\n\ndef test_square():\n"
                                        "    assert square_area(3) == 9\n")],
         check="python check.py"),
    Task("write_tests_clamp", "write_tests",
         "Write pytest tests in tests/test_clamp.py for clamp(x, lo, hi) in clamp.py, including values below, inside and "
         "above the range (at least 3 tests). The check also verifies your tests catch a bug in a mutated implementation.",
         {"clamp.py": "def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n",
          "mutants/clamp.py": "def clamp(x, lo, hi):\n    return max(lo, min(x, hi - 1))\n",
          "check.py": MUTANT_CHECK.format(min_tests=3, target="clamp.py", mutant="clamp.py"),
          "tests/__init__.py": ""},
         [write("tests/test_clamp.py", "from clamp import clamp\n\n\ndef test_below():\n    assert clamp(-5, 0, 10) == 0\n\n\n"
                                       "def test_inside():\n    assert clamp(5, 0, 10) == 5\n\n\ndef test_above():\n"
                                       "    assert clamp(50, 0, 10) == 10\n")],
         check="python check.py"),
]
