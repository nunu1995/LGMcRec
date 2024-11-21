import json
import openai


def generate_profiles(input_json, system_prompt_file, output_json):
     """
    Generate user/item profiles by calling OpenAI's GPT model.

    Args:
        input_json (str): Path to the input JSON file containing user prompts.
        introduction_prompt_file (str): Path to the introduction prompt file.
        output_json (str): Path to the output JSON file to save the generated profiles.

    Returns:
        None
    """

    with open(introduction_prompt_file, 'r') as f:
       introduction_prompt = f.read()

    with open(input_json, 'r') as f:
        data = json.load(f)
    example_prompts = [item['prompt'] for item in data]

    responses = []
    for idx, prompt in enumerate(example_prompts):
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": introduction_prompt},
                {"role": "user", "content": json.dumps(prompt)}
            ]
        ).choices[0].message.content

        try:
            response_json = json.loads(response)
            summarization = response_json.get("summarization", "")
        except json.JSONDecodeError:
            summarization = ""

        user_id = prompt[0]['UserID']
        responses.append({"UserID": user_id, "summarization": summarization})

    with open(output_json, 'w') as f:
        json.dump(responses, f, separators=(',', ':'))

