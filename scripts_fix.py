import re

# T1
f = 'atlas/problems/cec_pure/cec2017_pure.py'
with open(f, 'r') as fp: content = fp.read()
content = re.sub(r', aliases=\[\"cec2017_f\d+\"\]', '', content)
with open(f, 'w') as fp: fp.write(content)

# T12
f = 'atlas/core/experiment.py'
with open(f, 'r') as fp: content = fp.read()
content = content.replace('saver = ExperimentSaver(key, algo_name, cfg.base_dir)', 'saver = ExperimentSaver(key, algo_name, cfg.base_dir, run_dir=self.run_dir)')
content = content.replace('saver = ExperimentSaver(prob_key, label, cfg.base_dir)', 'saver = ExperimentSaver(prob_key, label, cfg.base_dir, run_dir=self.run_dir)')
content = content.replace('saver = ExperimentSaver(\"all_problems\", \"comparison\", cfg.base_dir)', 'saver = ExperimentSaver(\"all_problems\", \"comparison\", cfg.base_dir, run_dir=self.run_dir)')
content = content.replace('out_dir = Path(cfg.base_dir) / \"statistical_reports\"', 'out_dir = Path(self.run_dir) / \"statistical_reports\"')
with open(f, 'w') as fp: fp.write(content)
