import glob

f = 'tests/test_experiment.py'
with open(f, 'r', encoding='utf8') as fp: content = fp.read()
content = content.replace('stat_dir = tmp_path / "statistical_reports"', 'stat_dir = list(tmp_path.glob("experiment_*/statistical_reports"))[0]')
with open(f, 'w', encoding='utf8') as fp: fp.write(content)
