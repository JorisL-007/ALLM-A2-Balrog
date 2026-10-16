"""One E3 case; reuse the frozen E2 action loop with a composed reflection actor."""
import argparse
import json
import os
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parents[1]/'BALROG'))
import agents
import episode
from reflection import ReflectionAgent


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--spec',required=True)
    p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--reflection',action='store_true')
    a=p.parse_args()
    frozen=json.loads((HERE/'FROZEN.json').read_text())
    actor_variant=frozen['selected_actor']
    original=agents.make_agent
    instances=[]
    def make_agent(variant,factory,builder):
        actor=original(variant,factory,builder)
        if a.reflection:
            actor=ReflectionAgent(actor,json.loads(a.spec)['environment'],a.directory/'reflections.jsonl')
        instances.append(actor)
        return actor
    agents.make_agent=make_agent
    result=episode.run(json.loads(a.spec),actor_variant,a.directory,a.deadline)
    result.update(stage='E3',variant=actor_variant+('_reflection' if a.reflection else '_frozen'),
                  reflections=instances[0].reflections if a.reflection else 0)
    (a.directory/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
