from typing import List
import json
import yaml

class Location:
	def __init__(self,
				name, 
				contains=None, 
				agents=None):

		self.contains = contains if contains is not None else []
		self.agents = agents if agents is not None else [] # empty list to insert agents (ie str)
		self.name = name
		self.parent = None

		for one_contains in self.contains:
			for agent in one_contains.agents:
				if agent not in self.agents:
					self.agents.append(agent)
			one_contains.parent = self.name

		if name == "World":
			#can be used for easily locating nodes by their name
			self.name_to_node = {}
			self.dfs(self)


	def dfs(self, node):
		for node in node.contains:
			if node.name in self.name_to_node:
				raise ValueError("Duplicate location key name ", node.name)
			self.name_to_node[node.name] = node
			self.dfs(node)

		

	def to_json(self):
		info = {
			'name':self.name,
			'agents': self.agents,
			#'parent': self.parent,
			'contains': [item.to_json() for item in self.contains],
		}
		return info

	def add_agent(self, agent):
		self.agents.append(agent)
	
	def remove_agent(self, agent):
		self.agents.remove(agent)

	def add_contains(self, contains_item):
		self.contains.append(contains_item)
		contains_item.parent = self.name
	
	def remove_contains(self, contains_item):
		self.contains.remove(contains_item)
		contains_item.parent = None

	@classmethod
	def from_json_data(cls, json_data):

		return cls(
			name=json_data['name'],  
			agents=json_data['agents'],
			contains=[Location.from_json_data(element) for element in json_data['contains']]
		)

	def get_children_nl(self):
		children_list = [item.to_json()['name'] for item in self.contains]
		return ', '.join(children_list)



	@classmethod
	def from_yaml(self, filename):
		with open(filename, 'r') as file:
			loaded_yaml = yaml.safe_load(file)
		
		agent_to_location = {}
		for agent in loaded_yaml['Agents']:
			for key, value in agent.items():
				agent_to_location[key] = value

		loaded_yaml_map = {
			"name": "World",
			"agents": [],
			"contains": Location.convert_yaml_to_json(loaded_yaml['World'], agent_locations=agent_to_location)
		}
	
		loaded_location = Location.from_json_data(loaded_yaml_map)
		return loaded_location



	@staticmethod
	def convert_yaml_to_json(loaded_yaml, agent_locations=None):
		all_results = []
		for key in loaded_yaml:
			if isinstance(key, str):
				agents_at_curr_location = [name for name, loc in agent_locations.items() if loc == key]
				all_results.append({"name": key, "agents": agents_at_curr_location, "contains": []})
			# isinstance a dict
			else:
				for small_key in key:
					all_results.append({"name": small_key, "agents": [], "contains": Location.convert_yaml_to_json(key[small_key], agent_locations=agent_locations)})
		return all_results

