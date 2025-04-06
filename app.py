from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from preprocessor import preprocessor
from utils import (
    fetch_emoji_stats,
    most_common_words,
    generate_wordcloud,
    sentiment_analysis,
)
from pydantic import BaseModel
from typing import List
import base64
from io import BytesIO

app = FastAPI()

# Enable CORS so frontend can access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model for receiving a list of messages
class MessageList(BaseModel):
    messages: List[str]

# Upload and analyze chat
@app.post("/analyze")
async def analyze(file: UploadFile):
    content = await file.read()
    text = content.decode("utf-8")
    df = preprocessor(text)

    # Emoji stats
    emoji_df = fetch_emoji_stats(df['message'])

    # Word frequency
    common_words = most_common_words(df['message'], stopwords=[])

    # Wordcloud
    wc = generate_wordcloud(df['message'])
    buf = BytesIO()
    wc_img = wc.to_image()
    wc_img.save(buf, format="PNG")
    wordcloud_img = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Sentiment
    sentiments = sentiment_analysis(df['message'])

    # Timeline chart data
    df['date'] = df['date'].dt.date
    timeline = df.groupby('date').count()['message'].reset_index()
    timeline.columns = ['Date', 'Messages']
    timeline_data = timeline.to_dict(orient="records")

    return {
        "messages": df.to_dict(orient="records"),
        "emoji_stats": emoji_df.to_dict(orient="records"),
        "common_words": common_words.to_dict(orient="records"),
        "sentiments": sentiments,
        "wordcloud": wordcloud_img,
        "timeline": timeline_data
    }

# Optional endpoint for emoji stats (standalone)
@app.post("/emoji")
async def emoji_stats(data: MessageList):
    df = fetch_emoji_stats(data.messages)
    return df.to_dict(orient="records")
