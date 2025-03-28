from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI()


class CalendarEvent(BaseModel):
    reasoning: str = Field(description="The reasoning behind the LLM's response.")
    name: str
    date: str
    participants: list[str]


def main():
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "Extract the event information."},
            {"role": "user", "content": "AI learning seminar, October 20."},
        ],
        response_format=CalendarEvent,
    )

    print(completion.choices[0].message.parsed)


if __name__ == "__main__":
    main()
