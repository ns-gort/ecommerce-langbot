## Libraries ##

# Flask/Slack
from flask import Flask, Response, request, make_response
from slack import WebClient
from slackeventsapi import SlackEventAdapter

# Langchain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Util
from threading import Thread
import os
import yaml

# My QA-AI Code
import qa_ai


## Global Variables ##

app = Flask(__name__)

## Step 1: Open Config File ## 

with open('config.yaml', 'r') as config:
    config_file = yaml.safe_load(config)

## Step 2: Read Necessary API tokens ##

# Slack API Tokens

signing_secret = config_file["slack_secret"]
slack_oauthtoken = config_file["slack_oauthtoken"]
verification_token = config_file["verification_token"]

# OpenAI API token

os.environ['OPENAI_API_KEY'] = config_file["openai_token"]

## Step 3: Set up Slack interfaces

# Slack client 

slack_client = WebClient(slack_oauthtoken)

# Slack events adapter

slack_events_adapter = SlackEventAdapter(
    signing_secret, "/slack/events", app
)  

## Step 4: Read VectorStore data from disk

vector_store = FAISS.load_local(config_file["vectorstore"], OpenAIEmbeddings())

# Step 5: Set up QA-AI bot that answers queries

qa = qa_ai.QABot(vector_store)

## Flask App Endpoints ##

# This endpoint is used to verify the slack bot url

@app.route("/")
def event_hook(request):
    json_dict = json.loads(request.body.decode("utf-8"))
    if json_dict["token"] != verification_token:
        return {"status": 403}

    if "type" in json_dict:
        if json_dict["type"] == "url_verification":
            response_dict = {"challenge": json_dict["challenge"]}
            return response_dict
    return {"status": 500}
    return

# Call this endpoint on app_mention

@slack_events_adapter.on("app_mention")
def handle_message(event_data):
    def send_reply(value):

        event_data = value
        message = event_data["event"]
        
        if message.get("subtype") is None:
            command = message.get("text")
            channel_id = message["channel"]

            # Answer using AI

            response_to_query = qa.query_answer(command)

            slack_client.chat_postMessage(channel=channel_id, text=response_to_query)

    # Reply to message

    thread = Thread(target=send_reply, kwargs={"value": event_data})
    thread.start()
    return Response(status=200)

## Main Function: Run on startup ##

if __name__ == "__main__":
    ## Run Flask App

    app.run(port=3000)
