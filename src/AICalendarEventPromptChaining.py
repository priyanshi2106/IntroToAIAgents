import datetime
import json
import logging
import os
import warnings
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()  

import requests
from openai import OpenAI
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

# Suppress common deprecation warnings from dependencies
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pydantic')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='openai')

logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#define all the data models
#1. Event Extraction to check if the event is valid
class EventExtraction(BaseModel):
    ##First LLM call is made to this
    description: str = Field(description="The description of the event")
    is_calendar_event: bool = Field(description="Whether the description is a calendar event")
    confidence_score: float = Field(description="Confidence that this IS a calendar event (0.0-1.0). If is_calendar_event is True, this should be high (0.7-1.0). If is_calendar_event is False, this should be low (0.0-0.3)")

class EventDetails(BaseModel):
    #Second LLM call is made to this data model
    name: str = Field(description="The name of the event")
    date: str = Field(description="The date of the event. Use ISO format for this field")
    duration: int = Field(description="The duration of the event in minutes")
    participants: List[str] = Field(description="The participants of the event")

class EventConfirmation(BaseModel):
    #Third LLM call is made to this data model
    confirmation_message: str = Field(description="The confirmation message for the event")
    calendar_link: Optional[str] = Field(description="The calendar link for the event")

#Define the three functions that make call to LLM

def extractEventInfo(user_input: str) -> EventExtraction:
    """ First LLM call to determine if input is a calendar event or not"""
    logger.info("starting event extraction  ")
    
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    #adding date context to add it in prompt

    date_context = f"Today's date is {today}."
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"{date_context} Analyze if the text describes a calendar event.",
            },
            {
                "role": "user",
                "content": user_input,
            }
        ],
        response_format=EventExtraction,
    )
    result = completion.choices[0].message.parsed
    logger.info(f"Extraction complete - is calendar event: {result.is_calendar_event} with confidence score: {result.confidence_score} for input: {user_input}")
    return result

def parseEventDetails(description: str) -> EventDetails:
    """ Second LLM call to parse the event details"""
    logger.info("starting event details parsing")
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    date_context = f"Today's date is {today}."
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"{date_context} Analyze the user input and extract detailed the event details. When dates reference 'next Tuesday' or similar relative dates, use this current date as reference.",
            },
            {
                "role": "user",
                "content": description,
            }
        ],
        response_format=EventDetails,
    )
    result = completion.choices[0].message.parsed
    logger.info(f"Details parsed - name: {result.name}, date: {result.date}, duration: {result.duration}, participants: {result.participants}")
    return result  

def generateEventConfirmation(event_details: EventDetails) -> EventConfirmation:
    """ Third LLM call to generate the event confirmation"""
    logger.info("starting event confirmation generation")
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"Generate a natural confirmation message for the event. Sign off with your name: Priyanshi",
            },
            {
                "role": "user",
                "content": f"Generate a confirmation for this event: {event_details.name} on {event_details.date} for {event_details.duration} minutes with participants {', '.join(event_details.participants)}",
            }
        ],
        response_format=EventConfirmation,
    )
    result = completion.choices[0].message.parsed
    logger.info(f"Confirmation generated - confirmation message: {result.confirmation_message}, calendar link: {result.calendar_link}")
    return result



# Chaining the function together

def process_calendar_request(user_input: str) -> Optional[EventConfirmation]:
    """Main function implementing the prompt chaining with gate check"""
    logger.info("Starting calendar request processing")

    #extract basic info
    initial_extraction = extractEventInfo(user_input)
    # Gate check: If not a calendar event, return a generic response

    if(not initial_extraction.is_calendar_event or initial_extraction.confidence_score < 0.7):
        logger.warning("Gate check failed: Not a calendar event or confidence too low")
        return None
    logger.info("Gate check passed: Input is a calendar event")

    #Second LLM call
    event_details = parseEventDetails(initial_extraction.description)

    #Third LLM call
    confirmation = generateEventConfirmation(event_details)
    return confirmation

#Test it with valid input

user_input = "Can you schedule a meeting with Alice and Bob to discuss the project details for the new project?"
result = process_calendar_request(user_input)
if(result):
    print(f"Event confirmed: {result.confirmation_message}")
    if(result.calendar_link):
        print(f"Calendar link: {result.calendar_link}")
    else:
        print("No calendar link provided")
else:
    print("Event not confirmed as input might not be a calendar event")


#Test with invalid input

# user_input = "Can you send an email to Alice and Bob to discuss the project details for the new project?"
# result = process_calendar_request(user_input)
# if(result):
#     print(f"Event confirmed: {result.confirmation_message}")
#     if(result.calendar_link):
#         print(f"Calendar link: {result.calendar_link}")
#     else:
#         print("No calendar link provided")
# else:
#     print("Event not confirmed as input might not be a calendar event")