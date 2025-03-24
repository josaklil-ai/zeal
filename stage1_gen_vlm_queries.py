"""
Stage 1: VLM Query Generation

Use GPT4o or other proprietary/open-source LLMs to produce action descriptions 
and start/end questions to query a VLM agent for action localization in videos.
"""

import json
import openai
import os
import textwrap 

from dotenv import dotenv_values
from omegaconf import DictConfig
from tqdm import tqdm


LLM_PROMPT = {
    "SEQ": {
        "system": "You are an AI assistant that is part of a larger system designed to identify actions in videos.",
        # "user": textwrap.dedent("""
        #     For the following action, create a very short question about a key aspect of the very start 
        #     of the action and a very short question about the very end of the action. 
        #     Also, provide a short description of the action without adverbs like "forcefully" or "quickly". 
        #     The questions and description should be concise and should help differentiate between other action classes. 
        #     The questions should be yes/no answerable and answerable by observing only a single frame. That is, do not 
        #     give questions that require observing multiple frames. Also, make sure that the start question asks about something
        #     only unique to the beginning of the action, and make sure the end question asks about something only relevant to the
        #     end of the action. The description should be a concise summary of the action that will be used to retrieve relevant 
        #     frames from a video containing the action.
        #     Give your answer in a json that has the following structure:
        #     {
        #         "{{ACTION}}": {
        #             "start": <question>,
        #             "end": <question>,
        #             "description": <description>,
        #         },
        #     }
        #     The action is: {{ACTION}}
        # """)
        "user": textwrap.dedent("""
            Here is an action: <action>

            Please follow these steps to create a start question, end question, and description for the action:

            1. Consider what is unique about the very beginning of the action that could be observed in a single video frame. Craft a yes/no question about that.
            2. Consider what is unique about the very end of the action that could be observed in a single video frame. Craft a yes/no question about that. Make sure it does not overlap with the start question.
            3. Write a very short description summarizing the key components of the action, without using adverbs. The description should differentiate the action from other actions.

            Output your final answer in this JSON format:
            {
                "<action>": {
                    "start": "question",
                    "end": "question",
                    "description": "description"
                }
            }

            Make sure to follow the JSON formatting exactly, with the action in <action>. 
            Do not add any other elements to the JSON. Only include the start question, end question, and description.
            Do not include any explanatory text or preamble before the JSON. Only output the JSON.
        """)
    },
            
    "PHR": {
        "system": "You are a helpful AI assistant that is part of a larger system designed to identify actions in videos.",
        "user": textwrap.dedent("""
            You will be given an action, and your task is to come up with three distinct phrases that comprehensively 
            describe the key components or steps of that action. Here is the action to describe:
            <action>{{ACTION}}</action>
            To generate the three phrases, follow these steps:
            1. Carefully consider the action and break it down into its key physical components or motion sequences. Focus on aspects that would be visually apparent.
            2. Come up with three phrases that each capture a key component of the action. The phrases should be clearly distinct from one another, not overlapping in what they describe (mutually exclusive). At the same time, the three phrases together should cover all the essential elements of the action, not leaving any key parts out (collectively exhaustive).
            3. Phrase each component in a visually unambiguous way, as if you were describing it for a computer vision system to recognize. Avoid subjective or non-visual language like “very” or “extremely.”
            4. Keep each phrase concise, ideally just a few words capturing the essential physical motions or positions involved in that component of the action.
            Please repeat these steps to produce 3 phrases related to the action but would only help identify when the action is not occuring. 
            After generating all phrases, format your response in a json like this:
            {
                "{{ACTION}}": {
                    "pos": "First key component phrase, Second key component phrase, Third key component phrase",
                    "neg": "First phrase, Second phrase, Third phrase"
                },
            }
            Remember, the three positive and negative phrases should be visually distinct, cover all key components of the action, and be described in visually clear language.
        """)
    },

    "PHR-SEQ": {},
}


def gen_vlm_queries(
    cfg : DictConfig,
    llm_output_dir: str,
):
    env = dotenv_values()

    with open(os.path.join(os.getcwd(), "assets", f"{cfg.dataset.name}.json"), "r") as f:
        action_classes = list(json.load(f).values())

    client = openai.OpenAI(api_key=env.get("OPENAI_API_KEY"))

    if cfg.stage1.llm_type == "gpt4":
        model_name = "gpt-4-0613"
    elif cfg.stage1.llm_type == "gpt4o":
        model_name = "gpt-4o-2024-11-20"
    else:
        raise NotImplementedError

    block_content = []

    for action in tqdm(action_classes):
        completion = client.chat.completions.create(
                model=model_name,
                messages=[
                {"role": "system", "content": LLM_PROMPT[cfg.score_strategy]["system"]},
                {"role": "user", "content": LLM_PROMPT[cfg.score_strategy]["user"].replace("<action>", action)},
            ],
            max_tokens=cfg.stage1.max_tokens,
            temperature=cfg.stage1.temperature,
            response_format={"type": "json_object"},
        )
        
        content = json.loads(completion.choices[0].message.content)
        print(content)
        block_content.append(content)

    for block in block_content:
        content.update(block)

    with open(os.path.join(llm_output_dir, f"message_{cfg.score_strategy}.json"), 'w') as f:
        json.dump({"role": "assistant", "content": content}, f, indent=4)


if __name__ == "__main__":
    pass