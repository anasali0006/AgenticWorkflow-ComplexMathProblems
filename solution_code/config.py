from openai import OpenAI
from dotenv import load_dotenv
import os


def initialize_client():

    dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "OPENAI_KEY.env"))
    load_dotenv(dotenv_path)

    api_key = os.getenv("OPENAI_API_KEY")

    client = OpenAI(
        api_key=api_key,
    )

    return client