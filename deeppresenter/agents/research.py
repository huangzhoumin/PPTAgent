from deeppresenter.utils.typings import InputRequest

from .agent import Agent


class Research(Agent):
    async def loop(self, req: InputRequest):
        while True:
            print(f"req.attachments = {req.attachments}")
            print(f"req.deepresearch_prompt = {req.deepresearch_prompt}")
            agent_message = await self.action(
                prompt=req.deepresearch_prompt,
                attachments=req.attachments,
            )
            print(f"agent_message = {agent_message}")
            yield agent_message
            outcome = await self.execute(self.chat_history[-1].tool_calls)
            print(f"outcome = {outcome}")
            if isinstance(outcome, list):
                for item in outcome:
                    yield item
            else:
                yield outcome
                break
