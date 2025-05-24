import ollama

response = ollama.chat(
    model='llama3.2-vision',
    messages=[{
        'role': 'user',
        'content': 'Is the stove on?',
        'images': ['/home/arnav/Documents/coding/save-my-home/testing_data/prediction/IMG_2696.JPG']
    }]
)

print(response)