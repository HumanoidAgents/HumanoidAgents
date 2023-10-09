import json
import os
import argparse
from datetime import datetime
import logging

import yaml
from tqdm import tqdm

from humanoid_agent import HumanoidAgent

from location import Location
from utils import DatetimeNL, load_json_file, write_json_file, bucket_agents_by_location, get_pairwise_conversation_by_agents_in_same_location, override_agent_kwargs_with_condition

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


args = parser.parse_args()
logging.info(args)

folder_name = args.output_folder_name  
os.makedirs(folder_name, exist_ok=True)
map_filename = args.map_filename
agent_filenames = args.agent_filenames
condition = args.condition
start_date = args.start_date
end_date = args.end_date
default_agent_config_filename = args.default_agent_config_filename


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

## run simulation

for curr_date in dates_of_interest:
    curr_time = datetime.fromisoformat(curr_date)

    # plan at the start of day
    for agent in agents:
        agent.plan(curr_time=curr_time)

    for specific_time in tqdm(specific_times_of_interest):

        curr_time = datetime.fromisoformat(f'{curr_date}T{specific_time}')
        list_of_agent_statuses = []
        logging.info(curr_date + ' ' + specific_time)
        for agent in agents:
            overall_status = agent.get_status_json(curr_time, generative_location)
            list_of_agent_statuses.append(overall_status)

            logging.info("Overall status:")
            logging.info(json.dumps(overall_status, indent=4))

        location_to_agents = bucket_agents_by_location(list_of_agent_statuses, agents)
        location_to_conversations = get_pairwise_conversation_by_agents_in_same_location(location_to_agents, curr_time)
        for location, conversations in location_to_conversations.items():
            logging.info(f"Conversations at {location}")
            logging.info(json.dumps(conversations, indent=4))
        
        overall_log = {
            "date": DatetimeNL.get_date_nl(curr_time),
            "time": DatetimeNL.get_time_nl(curr_time),
            "agents": list_of_agent_statuses,
            "conversations": {'-'.join(location): conversations for location, conversations in location_to_conversations.items()},
            "world": generative_location.to_json()
        }
        output_filename = f"{folder_name}/state_{curr_date}_{specific_time.replace(':','h')}.json"
        write_json_file(overall_log, output_filename)
