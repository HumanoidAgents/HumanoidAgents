import json
import os
import argparse
from datetime import datetime
import logging
import random
from collections import defaultdict
from urllib.parse import urljoin

from humanoid_agent import HumanoidAgent
from location import Location
from utils import DatetimeNL, load_json_file, write_json_file, bucket_agents_by_location, override_agent_kwargs_with_condition, get_curr_time_to_daily_event

import requests
from flask import Flask, request, url_for, request
from flask_caching import Cache


# server side caching
config = {
    "DEBUG": True,          # some Flask specific configs
    "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300
}
app = Flask(__name__)
app.config.from_mapping(config)
cache = Cache(app)

@app.route('/')
def index():
    return 'hi'

@app.route('/chat_single_turn', methods=['POST', 'GET'])
@cache.cached(query_string=True)
def chat_single_turn_with_possible_responses(n_responses=3):
    data = parse_request(request, expected_keys=['curr_date', 'specific_time', 'initiator_name', 'responder_name', 'conversation_history'])

    # this is an error message
    if isinstance(data, str):
        return data

    curr_date = data['curr_date']
    specific_time = data['specific_time']
    initiator_name = data['initiator_name']
    responder_name = data['responder_name']
    conversation_history = data['conversation_history']

    curr_time = datetime.fromisoformat(f'{curr_date}T{specific_time}')

    initiator = name_to_agent[initiator_name]
    responder = name_to_agent[responder_name]

    possible_responses = []
    for i in range(n_responses):
        reaction = responder.get_agent_reaction_about_another_agent(initiator, curr_time, conversation_history=conversation_history)
        response = responder.speak_to_other_agent(initiator, curr_time, reaction=reaction, conversation_history=conversation_history)
        possible_responses.append(response)

    return possible_responses

#need variable number of messages thorugh POST + give users options + user defines what to use 
@app.route('/chat', methods=['POST', 'GET'])
@cache.cached(query_string=True)
def chat():

    data = parse_request(request, expected_keys=['curr_date', 'specific_time', 'initiator_name', 'responder_name'])

    # this is an error message
    if isinstance(data, str):
        return data

    curr_date = data['curr_date']
    specific_time = data['specific_time']
    initiator_name = data['initiator_name']
    responder_name = data['responder_name']

    curr_time = datetime.fromisoformat(f'{curr_date}T{specific_time}')

    initiator = name_to_agent[initiator_name]
    responder = name_to_agent[responder_name]
    convo_history = initiator.dialogue(responder, curr_time)

    return convo_history

def initialize(args):
    folder_name = args.output_folder_name  
    os.makedirs(folder_name, exist_ok=True)
    map_filename = args.map_filename
    agent_filenames = args.agent_filenames
    condition = args.condition
    start_date = args.start_date
    end_date = args.end_date
    default_agent_config_filename = args.default_agent_config_filename
    llm_provider = args.llm_provider
    llm_model_name = args.llm_model_name
    embedding_model_name = args.embedding_model_name
    daily_events_filename = args.daily_events_filename

    ## location
    generative_location = Location.from_yaml(map_filename)

    ## agents
    agents = []

    default_agent_kwargs = load_json_file(default_agent_config_filename)

    for agent_filename in agent_filenames:
        agent_kwargs = load_json_file(agent_filename)
        #inplace dict update
        agent_kwargs.update(default_agent_kwargs)
        agent_kwargs = override_agent_kwargs_with_condition(agent_kwargs, condition)
        agent_kwargs["llm_provider"] = llm_provider
        agent_kwargs["llm_model_name"] = llm_model_name
        agent_kwargs["embedding_model_name"] = embedding_model_name
        agent = HumanoidAgent(**agent_kwargs)
        agents.append(agent)

    ## time
    dates_of_interest = DatetimeNL.get_date_range(start_date, end_date)
    specific_times_of_interest = []
    for hour in range(6, 24):
        for minutes in ['00', '15', '30', '45']:
            hour_str = str(hour) if hour > 9 else '0' + str(hour)
            total_time = f"{hour_str}:{minutes}"
            specific_times_of_interest.append(total_time)
    
    ## daily_events
    curr_time_to_daily_event = get_curr_time_to_daily_event(daily_events_filename)

    return agents, dates_of_interest, specific_times_of_interest, generative_location, folder_name, curr_time_to_daily_event 

def parse_request(request, expected_keys=['curr_date', 'specific_time']):

    key_to_format = {
        'curr_date': "yyyy-mm-dd",
        'specific_time': "hh:mm",
        'name': "FirstName LastName",
        'conversation_history': 'list of { "name": name_self, "text": speak_self, "reaction": response_self}'
    }

    key_to_example = {
        'curr_date': "2023-01-03", 
        'specific_time': "09:00",
        'name': "John Lin",
        'conversation_history': '[{ "name": "John Lin", "text": "Hi there", "reaction": "Welcome Eddy back"}]'
    }

    key_to_options = {
        'name': list(name_to_agent.keys())
    }
    # automatically detect if data in GET or POST request form
    app_data = request.args if request.args else request.json

    # parameter validation
    useful_data = {}

    for key in expected_keys:
        if 'name' in key:
            short_key = 'name'
        else:
            short_key = key

        format = key_to_format[short_key]
        example = key_to_example[short_key]

        if key not in app_data:
            return f"{key} with required format {format} (e.g {example}) not in request"

        if short_key in key_to_options and app_data[key] not in key_to_options[short_key]:
            return f"For {key}, choose only from this list {key_to_options[short_key]}"
        useful_data[key] = app_data[key]
    
    return useful_data

@app.route('/plan_single', methods=['POST', 'GET'])
@cache.cached(query_string=True)
def plan_single():

    data = parse_request(request, expected_keys=['curr_date', 'name'])
    
    # this is an error message
    if isinstance(data, str):
        return data
    curr_date = data['curr_date']
    name = data['name']

    curr_time = datetime.fromisoformat(curr_date)
    condition = curr_time_to_daily_event[curr_time] if curr_time in curr_time_to_daily_event else None
    plan = name_to_agent[name].plan(curr_time=curr_time, condition=condition)
    return plan 

@app.route('/plan', methods=['POST', 'GET'])
def plan():
    data = parse_request(request, expected_keys=['curr_date'])
    
    # this is an error message
    if isinstance(data, str):
        return data
    curr_date = data['curr_date']

    plans = []
    for agent in agents:
        plan = requests.get(
            url=urljoin(request.base_url, url_for("plan_single")), 
            params = {
                "curr_date": curr_date,
                "name": agent.name
        }).text
        plans.append(plan)
    return plans

@app.route('/activity_single', methods=['POST', 'GET'])
@cache.cached(query_string=True)
def get_15m_activity_single():

    data = parse_request(request, expected_keys=['curr_date', 'specific_time', 'name'])

    # this is an error message
    if isinstance(data, str):
        return data

    curr_date = data['curr_date']
    specific_time = data['specific_time']
    name = data['name']

    curr_time = datetime.fromisoformat(f'{curr_date}T{specific_time}')

    logging.info(curr_date + ' ' + specific_time)

    overall_status = name_to_agent[name].get_status_json(curr_time, generative_location)
    logging.info("Overall status:")
    logging.info(json.dumps(overall_status, indent=4))
    return overall_status

@app.route('/activity', methods=['POST', 'GET'])
@cache.cached(query_string=True)
def get_15m_activity():

    data = parse_request(request, expected_keys=['curr_date', 'specific_time'])

    # this is an error message
    if isinstance(data, str):
        return data

    curr_date = data['curr_date']
    specific_time = data['specific_time']

    # curr_time = datetime.fromisoformat(f'{curr_date}T{specific_time}')
    list_of_agent_statuses = []
    logging.info(curr_date + ' ' + specific_time)
    for agent in agents:
        overall_status = requests.get(
            url=urljoin(request.base_url, url_for("get_15m_activity_single")), 
            params = {
                "curr_date": curr_date,
                "name": agent.name,
                "specific_time": specific_time
        }).json()
        
        # agent.get_status_json(curr_time, generative_location)
        list_of_agent_statuses.append(overall_status)

        logging.info("Overall status:")
        logging.info(json.dumps(overall_status, indent=4))
    return list_of_agent_statuses

@app.route('/conversations', methods=['POST', 'GET'])
@cache.cached(query_string=True)
def get_15m_conversations():

    data = parse_request(request, expected_keys=['curr_date', 'specific_time'])

    # this is an error message
    if isinstance(data, str):
        return data

    curr_date = data['curr_date']
    specific_time = data['specific_time']
    list_of_agent_statuses = requests.get(
        url=urljoin(request.base_url, url_for("get_15m_activity")), 
        params = {
            "curr_date": curr_date,
            "specific_time": specific_time
    }).json()

    global agents
    location_to_agents = bucket_agents_by_location(list_of_agent_statuses, agents)
    # only 1 conversation per location, when there are 2 or more agents
    location_to_conversations = defaultdict(list)
    for location, agents in location_to_agents.items():
        if len(agents) > 1:
            selected_agents = random.sample(agents, 2)
            initiator, responder = selected_agents
            convo_history = requests.get(
                url=urljoin(request.base_url, url_for("chat")), 
                params = {
                    "curr_date": curr_date,
                    "specific_time": specific_time,
                    "initiator_name": initiator.name,
                    "responder_name": responder.name
            }).json()
            logging.info(f"Conversations at {location}")
            logging.info(json.dumps(convo_history, indent=4))
            location_to_conversations['-'.join(location)].append(convo_history)
    return location_to_conversations

@app.route('/logs', methods=['POST', 'GET'])
def write_logs():

    data = parse_request(request, expected_keys=['curr_date', 'specific_time'])

    # this is an error message
    if isinstance(data, str):
        return data

    curr_date = data['curr_date']
    specific_time = data['specific_time']
    list_of_agent_statuses = requests.get(
        url=urljoin(request.base_url, url_for("get_15m_activity")), 
        params = {
            "curr_date": curr_date,
            "specific_time": specific_time
    }).json()

    location_to_conversations = requests.get(
        url=urljoin(request.base_url, url_for("get_15m_conversations")), 
        params = {
            "curr_date": curr_date,
            "specific_time": specific_time
    }).json()

    overall_log = {
        "date": curr_date,
        "time": specific_time,
        "agents": list_of_agent_statuses,
        "conversations": {location: conversations for location, conversations in location_to_conversations.items()},
        "world": generative_location.to_json()
    }
    output_filename = f"{folder_name}/state_{curr_date}_{specific_time.replace(':','h')}.json"
    write_json_file(overall_log, output_filename)
    return overall_log

if __name__ == '__main__':
    logging.basicConfig(format='---%(asctime)s %(levelname)s \n%(message)s ---', level=logging.INFO)

    parser = argparse.ArgumentParser(description='run humanoid agents simulation')
    parser.add_argument("-o", "--output_folder_name", required=True)
    parser.add_argument("-m", "--map_filename", required=True) # '../locations/lin_family_map.yaml'
    parser.add_argument("-a", "--agent_filenames", required=True, nargs='+') # "../specific_agents/john_lin.json", "../specific_agents/eddy_lin.json"
    parser.add_argument("-da", "--default_agent_config_filename", default="default_agent_config.json")
    parser.add_argument('-s', '--start_date', help='Enter start date (inclusive) by YYYY-MM-DD e.g.2023-01-03', default="2023-01-03")
    parser.add_argument('-e', '--end_date', help='Enter end date (inclusive) by YYYY-MM-DD e.g.2023-01-04', default="2023-01-03")
    parser.add_argument("-c", "--condition", default=None, choices=["disgusted", "afraid", "sad", "surprised", "happy", "angry", "neutral", 
                                            "fullness", "social", "fun", "health", "energy", 
                                            "closeness_0", "closeness_5", "closeness_10", "closeness_15", None])
    parser.add_argument("-l", "--llm_provider", default="openai", choices=["openai", "local", "mindsdb"])
    parser.add_argument("-lmn", "--llm_model_name", default="gpt-3.5-turbo")
    parser.add_argument("-emn", "--embedding_model_name", default="text-embedding-ada-002", help="with local, please use all-MiniLM-L6-v2 or another name compatible with SentenceTransformers")
    parser.add_argument("-daf", "--daily_events_filename", default=None)


    args = parser.parse_args()
    logging.info(args)
    agents, dates_of_interest, specific_times_of_interest, generative_location, folder_name, curr_time_to_daily_event = initialize(args)
    name_to_agent = {agent.name: agent for agent in agents}
    print("starting")
    app.run(debug=True)
    

    


