


def get_chat_completion(client, model, system_message, user_message):

    completion = client.chat.completions.create(
    model = model, 
    messages=[
        {"role": "developer", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    )

    return completion.choices[0].message


def get_json_completion(client, model, system_message, user_message, format):
    
    completion = client.beta.chat.completions.parse(
    model=model,
    messages=[
        {"role": "developer", "content": system_message},
        {"role": "user", "content": user_message}
    ],
    response_format=format
    )

    return completion.choices[0].message.parsed


def get_embedding(client, text, model):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding