import pandas as pd
import re

def preprocessor(data):
    # Pattern to match WhatsApp timestamp (supports optional space before am/pm)
    pattern = r'\d{2}/\d{2}/\d{4}, \d{1,2}:\d{2}\s?[ap]m\s?-'

    # Split messages and extract date strings
    messages = re.split(pattern, data)[1:]  # First split is before the first message
    dates = re.findall(pattern, data)

    # Clean and convert date strings
    clean_dates = [d.replace('\u202f', ' ').strip() for d in dates]  # Normalize any Unicode narrow spaces

    # Create dataframe
    df = pd.DataFrame({'user_message': messages, 'message_date': clean_dates})
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%Y, %I:%M %p -')
    df.rename(columns={'message_date': 'date'}, inplace=True)

    # Separate users and messages
    users = []
    messages_cleaned = []

    for message in df['user_message']:
        entry = re.split(r'([\w\W]+?):\s', message)
        if len(entry) > 2:  # Message from user
            users.append(entry[1])
            messages_cleaned.append(entry[2])
        else:  # System message or group notification
            users.append('group_notification')
            messages_cleaned.append(entry[0])

    df['user'] = users
    df['message'] = messages_cleaned
    df.drop(columns=['user_message'], inplace=True)

    # Add time-based features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    df['month_num'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    return df
