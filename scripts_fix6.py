import re

f = 'tests/test_experiment.py'
with open(f, 'r', encoding='utf8') as fp: content = fp.read()
content = content.replace('stat_dir = Path(tmpdir) / "statistical_reports"', 'stat_dir = list(Path(tmpdir).glob("experiment_*/statistical_reports"))[0]')
with open(f, 'w', encoding='utf8') as fp: fp.write(content)
