"""Optional within-episode reflection, inspired by course book §10.4/Reflexion.

This is a bounded online correction pilot, not a reproduction of cross-trial
Reflexion. Actor is frozen first. No lessons survive an episode boundary.
"""
import copy
import json
import time
from pathlib import Path


class ReflectionAgent:
    def __init__(self, actor, environment, log_path, threshold=3, cooldown=12, max_reflections=2):
        self.actor=actor
        self.prompt_builder=actor.prompt_builder
        self.client=actor.client
        self.environment=environment
        self.log_path=Path(log_path)
        self.threshold,self.cooldown,self.max_reflections=threshold,cooldown,max_reflections
        get_prompt=self.prompt_builder.get_prompt
        def with_lessons(*args,**kwargs):
            messages=copy.deepcopy(get_prompt(*args,**kwargs))
            if self.lessons:
                messages[-1].content+='\n\nWithin-episode corrective lessons (check against current observation):\n'+'\n'.join(self.lessons)
            return messages
        self.prompt_builder.get_prompt=with_lessons
        self.reset()

    def reset(self):
        self.actor.reset()
        self.lessons=[]
        self.steps=self.streak=self.reflections=0
        self.last_reflection=-self.cooldown
        self.last_action=None
        self.trigger_context=None

    def observe_transition(self,action,before,after):
        # Do/Sleep/Noop/idle can be meaningful despite little visible spatial change.
        exempt=action in {'Do','Sleep','Noop','idle'}
        same=before==after
        self.streak=self.streak+1 if same and not exempt and action==self.last_action else (1 if same and not exempt else 0)
        self.last_action=action
        self.trigger_context=dict(action=action,before=before,after=after,unchanged_streak=self.streak)
        if hasattr(self.actor,'observe_transition'):
            self.actor.observe_transition(action,before,after)

    def act(self,obs,prev_action=None):
        extra_input=extra_output=0
        if self.streak>=self.threshold and self.steps-self.last_reflection>=self.cooldown and self.reflections<self.max_reflections:
            from balrog.prompt_builder.history import Message
            prompt=('Review this recent action-effect mismatch using only the provided public text. '
                    'The same action left the exposed text unchanged repeatedly. That is a heuristic, '
                    'not proof of failure. Give one concise corrective lesson for later decisions, '
                    'not an action or a multi-step solution. Do not invent hidden state.\n'
                    +'Public game instructions:\n'+str(self.prompt_builder.system_prompt)+'\n'
                    +json.dumps(self.trigger_context,ensure_ascii=False))
            old_budget=self.client.tokens
            reflection_started=time.monotonic()
            try:
                self.client.tokens=96
                response=self.client.generate([Message('user',prompt)])
            finally:
                self.client.tokens=old_budget
            lesson=(response.completion or '').strip()[:400]
            if lesson:
                self.lessons=(self.lessons+[lesson])[-2:]
            self.reflections+=1
            self.last_reflection=self.steps
            extra_input=response.input_tokens or 0
            extra_output=response.output_tokens or 0
            self.log_path.parent.mkdir(parents=True,exist_ok=True)
            with self.log_path.open('a') as f:
                f.write(json.dumps(dict(step=self.steps,trigger=self.trigger_context,prompt=prompt,
                    response=response.completion,lesson=lesson,memory=list(self.lessons),
                    input_tokens=extra_input,output_tokens=extra_output,stop_reason=response.stop_reason,
                    elapsed_seconds=time.monotonic()-reflection_started,
                    call_id=getattr(self.client,'count',0)-1))+'\n')
        response=self.actor.act(obs,prev_action=prev_action)
        self.steps+=1
        return response._replace(input_tokens=(response.input_tokens or 0)+extra_input,
                                 output_tokens=(response.output_tokens or 0)+extra_output)
