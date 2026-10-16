#!/usr/bin/env python3
"""Load our CustomAgent for this process without editing the BALROG checkout."""
import argparse
import ast
import os
import runpy
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--balrog-dir', type=Path, required=True)
    parser.add_argument('--variant', choices=('planning', 'grounded'), required=True)
    parser.add_argument('--environment', choices=('babyai', 'crafter', 'babaisai'), required=True)
    parser.add_argument('--log-dir', type=Path, required=True)
    args, overrides = parser.parse_known_args()
    if overrides[:1] == ['--']:
        overrides = overrides[1:]
    repo = args.balrog_dir.resolve()
    if not (repo / 'eval.py').is_file():
        parser.error('BALROG eval.py not found')
    os.environ.update(EX2_VARIANT=args.variant, EX2_ENVIRONMENT=args.environment,
                      EX2_LOG_DIR=str(args.log_dir.resolve()))
    sys.path.insert(0, str(repo))
    import balrog.agents as agents
    from ex2_agent import ACTIONS, CustomAgent
    # Refuse a silently incompatible action set after an upstream update.
    sources = {
        'babyai': ('babyai_text/clean_lang_wrapper.py', 'BABYAI_ACTION_SPACE'),
        'crafter': ('crafter/env.py', 'ACTIONS'),
        'babaisai': ('babaisai/__init__.py', 'ACTIONS'),
    }
    relative, name = sources[args.environment]
    tree = ast.parse((repo / 'balrog/environments' / relative).read_text())
    value = next(ast.literal_eval(n.value) for n in tree.body
                 if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    if set(value) != set(ACTIONS[args.environment]):
        parser.error('BALROG action set changed; update ex2_agent.py before running')
    agents.CustomAgent = CustomAgent
    os.chdir(repo)
    sys.argv = [str(repo / 'eval.py'), *overrides,
                'agent.type=custom', f'envs.names={args.environment}']
    runpy.run_path(str(repo / 'eval.py'), run_name='__main__')


if __name__ == '__main__':
    main()
