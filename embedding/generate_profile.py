import json
import openai


def generate_profiles(input_json, system_prompt_file, output_json):

    with open(system_prompt_file, 'r') as f:
        system_prompt = f.read()

    with open(input_json, 'r') as f:
        data = json.load(f)
    example_prompts = [item['prompt'] for item in data]

    responses = []
    for idx, prompt in enumerate(example_prompts):
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": system_prompt},
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

