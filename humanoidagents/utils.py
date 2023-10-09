import json
import random
from collections import defaultdict
from datetime import datetime, timedelta

def load_json_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_json_file(data, filename):
    with open(filename, "w", encoding="utf-8") as fw:
        fw.write(json.dumps(data, indent=4))

def bucket_agents_by_location(list_of_agent_statuses, agents):
    location_to_agents = defaultdict(list)
    for agent_status, agent in zip(list_of_agent_statuses, agents):
        agent_location = agent_status['location']
        # to make agent location hashable
        tuple_agent_location = tuple(agent_location)
        location_to_agents[tuple_agent_location].append(agent)
    return location_to_agents

def get_pairwise_conversation_by_agents_in_same_location(location_to_agents, curr_time):
    # only 1 conversation per location, when there are 2 or more agents

    location_to_conversations = defaultdict(list)
    for location, agents in location_to_agents.items():
        if len(agents) > 1:
            selected_agents = random.sample(agents, 2)
            initiator, responder = selected_agents
            convo_history = initiator.dialogue(responder, curr_time)
            location_to_conversations[location].append(convo_history)
    return location_to_conversations

def override_agent_kwargs_with_condition(kwargs, condition):
    if condition is None:
        return kwargs
    #emotion
    elif condition in ["disgusted", "afraid", "sad", "surprised", "happy", "angry", "neutral"]:
        kwargs["emotion"] = condition
    #set each social relationship closeness to specific value
    elif "closeness" in condition:
        for key in kwargs["social_relationships"]:
            kwargs["social_relationships"][key]['closeness'] = int(condition.split("_")[1])
    #basic needs
    elif condition in ["fullness", "social", "fun", "health", "energy"]:
        if "basic_needs" not in kwargs:
            kwargs["basic_needs"] = {}
        for condition_i, basic_need in enumerate(kwargs["basic_needs"]):
            if basic_need['name'] == condition:
                kwargs["basic_needs"][condition_i]["start_value"] = 0
    else:
        raise ValueError("condition is not valid")
    return kwargs

class DatetimeNL:

    @staticmethod
    def get_date_nl(curr_time):
        # e.g. Monday Jan 02 2023
        day_of_week = curr_time.strftime('%A')
        month_date_year = curr_time.strftime("%b %d %Y")
        date = f"{day_of_week} {month_date_year}"
        return date

    @staticmethod
    def get_time_nl(curr_time):
        #e.g. 12:00 am
        time = curr_time.strftime('%I:%M %p').lower()
        if time.startswith('0'):
            time = time[1:]
        return time

    @staticmethod
    def convert_nl_datetime_to_datetime(date, time):
        # missing 0 in front of time
        if len(time) != len("12:00 am"):
            time = "0" + time.upper()
        
        concatenated_date_time = date + ' ' + time
        curr_time = datetime.strptime(concatenated_date_time, "%A %b %d %Y %I:%M %p")
        return curr_time

    @staticmethod
    def get_formatted_date_time(curr_time):
        # e.g. "It is Monday Jan 02 2023 12:00 am"
        date_in_nl = DatetimeNL.get_date_nl(curr_time)
        time_in_nl = DatetimeNL.get_time_nl(curr_time)
        formatted_date_time = f"It is {date_in_nl} {time_in_nl}"
        return formatted_date_time
    
    @staticmethod
    def get_date_range(start_date, end_date):
        """
        Get date range between start_date (inclusive) and end_date (inclusive)
        
        start_date and end_date are str in the format YYYY-MM-DD
        """
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        date_range = []
        
        while start_date <= end_date:
            date_range.append(start_date.strftime('%Y-%m-%d'))
            start_date += timedelta(days=1)
        if not date_range:
            raise ValueError("end_date must be later or equal to start_date")
        return date_range
