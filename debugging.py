from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_name = "nateraw/bert-base-uncased-emotion"  # Public model

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create emotion classifier pipeline
emotion_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

gif_map = {
    "neutral:hello.gif"
    "anger": "upset.gif",
    "Anger":"anger.gif",
    "annoyance": "frustrated.gif",
    "disgust": "upset.gif",
    "fear": "panic.gif",
    "joy": "Greet.gif",
    "neutral": "ok.gif",
    "sadness": "hurt.gif",
    "surprise": "thinking.gif",
    "amusement": "chill.gif",
    "approval": "ok.gif",
    "caring": "friendship.gif",
    "confusion": "thinking.gif",
    "curiosity": "thinking.gif",
    "desire": "power.gif",
    "disappointment": "upset.gif",
    "disapproval": "anger.gif",
    "embarrassment": "thinking.gif",
    "excitement": "power.gif",
    "gratitude": "friendship.gif",
    "grief": "hurt.gif",
    "guilt": "hurt.gif",
    "hope": "ok.gif",
    "love": "friendship.gif",
    "melancholy": "hurt.gif",
    "nervousness": "panic.gif",
    "nostalgia": "thinking.gif",
    "optimism": "power.gif",
    "pride": "power.gif",
    "realization": "thinking.gif",
    "relief": "ok.gif",
    "remorse": "hurt.gif",
    "resignation": "sleepy.gif",
    "shame": "thinking.gif",
    "tiredness": "sleepy.gif",
    "sympathy": "friendship.gif",
    "boredom": "chilling.gif",
    "loneliness": "hurt.gif",
    "envy": "thinking.gif",
    "frustration": "frustrated.gif",
    "contentment": "chill.gif",
    "inspiration": "power.gif",
    "indifference": "ok.gif",
    "betrayal": "hurt.gif",
    "exhaustion": "sleepy.gif",
    "panic": "panic.gif",
    "conflicted": "thinking.gif",
    "determination": "power.gif",
    "sarcasm": "thinking.gif",
    "triumph": "power.gif",
    "hunger": "hunger.gif",
    "thirsty": "thirsty.gif",
}

def detect_emotion(sentence):
    result = emotion_classifier(sentence)
    emotion = result[0]['label']
    return emotion

def get_gif_for_emotion(emotion):
    return gif_map.get(emotion, "ok.gif")  # Default to "ok.gif" if no match is found




