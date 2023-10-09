import json
from datetime import datetime
from functools import cache
import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils import DatetimeNL
from llm import OpenAILLM

class GenerativeAgent:

    def __init__(self, name: str, description: str, age: int, traits: list, example_day_plan: str):
        self.name = name
        self.description = description
        self.age = age
        self.traits = traits
        self.example_day_plan = example_day_plan if isinstance(example_day_plan, str) else '\n'.join(example_day_plan)

        # the actual game starts on 2023-01-03, so setting provided info as the day before
        global_start_date = datetime.fromisoformat('2023-01-02')

        statements = description.split(';') if isinstance(description, str) else description

        self.memory = []

        for statement in statements:
            self.add_to_memory(
                activity=statement.strip(), 
                curr_time=global_start_date,
                memory_type="provided_statements"
            )
        
        self.add_to_memory(activity=self.example_day_plan, curr_time=global_start_date, memory_type="day_plan")

    @cache
    def plan(self, curr_time, condition=None):
        plan = self.initial_plan(curr_time, condition=condition)
        logging.info("day_plan")
        logging.info(plan)
        hourly_plan = self.recursively_decompose_plan(plan, curr_time=curr_time , time_interval="1 hour")
        logging.info("hourly_plan")
        logging.info(hourly_plan)
        minute_15_plan_based_on_hourly = self.recursively_decompose_plan(hourly_plan, curr_time=curr_time, time_interval="15 minutes")
        logging.info("minute_15_plan_based_on_hourly")
        logging.info(minute_15_plan_based_on_hourly)

    @staticmethod
    def check_plan_format(plan):
        if plan is None:
            return False

        plan_items = plan.split('\n')

        #check for overly short plans
        if len(plan_items) < 2:
            return False 
        #print(json.dumps(plan_items, indent=4))
        for plan_item in plan_items:
            # every plan_item is in the format hh:mm pm/am or h:am pm/am
            plan_colon_split = plan_item.split(":")
            # starts with a time
            if len(plan_colon_split) > 2 and plan_colon_split[0].isdigit() and len(plan_colon_split[1].split(" ")) == 2 and plan_colon_split[1].split(" ")[0].isdigit() and plan_colon_split[1].split(" ")[1] in ["am", "pm"]:
                pass
            else:
                return False
        plan_item = plan_items[-1]
        
        #we want a plan that ends before midnight
        if plan_item.split(":")[1].split(" ")[1] == "am":
            return False

        return True

    def initial_plan(self, curr_time, condition=None, max_attempts=10):
        """
        This creates a daily plan using a person's name, age, traits, and a self description and the latest plan
        """
        date = DatetimeNL.get_date_nl(curr_time)

        last_plan_memory = [memory_item for memory_item in self.memory if memory_item['memory_type'] == 'day_plan'][-1]
        last_plan_activity = last_plan_memory['activity']
        last_plan_time = last_plan_memory['creation_time']
        last_plan_time_nl = DatetimeNL.get_date_nl(last_plan_time)

        condition_formatted = f'\nCondition: {condition}\n' if condition is not None else ''

        prompt = f"""
        Please plan a day for {self.name} ending latest by 11:45 pm.

        Format:
        hh:mm am/pm: <activity>

        Name: {self.name} (age: {self.age})
        Innate traits: {', '.join(self.traits)}
        Description: {self.description}{condition_formatted}
        On {last_plan_time_nl}, 
        {last_plan_activity}
        On {date},
        """

        #for re-planning from a certain time (e.g. 4:00 pm) onwards
        if DatetimeNL.get_time_nl(curr_time) != "12:00 am":
            prompt += f"{DatetimeNL.get_time_nl(curr_time)}:"

        
        resulting_plan = None
        
        attempts = 0
        while not GenerativeAgent.check_plan_format(resulting_plan) and attempts < max_attempts:
            resulting_plan = OpenAILLM.get_llm_response(prompt)
            if DatetimeNL.get_time_nl(curr_time) != "12:00 am":
                resulting_plan = f"{DatetimeNL.get_time_nl(curr_time)}:" + resulting_plan
            resulting_plan = resulting_plan.split('\n')
            resulting_plan = '\n'.join([plan_item for plan_item in resulting_plan if plan_item.strip()])
            attempts += 1
            logging.info(f"planning day attempt number {attempts} / {max_attempts}")
            logging.info(resulting_plan)

        if attempts == max_attempts:
            raise ValueError("Initial Plan generation failed")

        self.add_to_memory(activity=resulting_plan, curr_time=curr_time, memory_type="day_plan")
        return resulting_plan

    def add_to_memory(self, activity, curr_time, calculate_importance=False, **kwargs):
        memory_item = {
            "creation_time": curr_time, 
            "activity": activity, 
            "last_access_time": curr_time,
            "importance": 5 if not calculate_importance else GenerativeAgent.calculate_importance(activity),
        }
        for arg in kwargs:
            memory_item[arg] = kwargs[arg]

        self.memory.append(memory_item)

    def recursively_decompose_plan(self, plan, curr_time, time_interval="1 hour", max_attempts=10):
        ## using hourly plan (instead of whole plan) to obtain 15 minute plan can lead to more detailed plans, with less skipped intervals)

        # note we can even plan entire weeks or months or years with this recusive strategy
        prompt = f"""
        Please decompose the plan into items at intervals of {time_interval}, ending the day latest by 11:45 pm.
        Format: hh:mm am/pm: <activity>

        Plan: 
        6:00 am: woke up and completed the morning routine
        7:00 am: finished breakfast
        8:00 am: opened up The Willows Market and Pharmacy
        8:30 am: greeted the regular customers and helped them with their medication needs
        12:00 pm: had lunch
        1:00 pm: continued working and assisting customers
        7:00 pm: closed up the shop and went home 
        8:00 pm: have dinner with his family
        9:00 pm: watched a movie with his son, Eddy
        10:00 pm: get ready for bed and slept

        Plan in intervals of 1 hour: 
        6:00 am: woke up and completed the morning routine 
        7:00 am: finished breakfast 
        8:00 am: opened up The Willows Market and Pharmacy 
        9:00 am: greeted the regular customers 
        10:00 am: helped regular customers with their medication needs 
        11:00 am: greeted more customers 
        12:00 pm: had lunch 
        1:00 pm: restocked medication 
        2:00 pm: checked computers on medications he should order
        3:00 pm: checked shelves to see whether popular medications are still in stock
        4:00 pm: helped with prescription of customers
        5:00 pm: helped with prescription of customers
        6:00 pm: helped customers with questions about side effects of medication
        7:00 pm: closed shop and went home
        8:00 pm: had dinner with family
        9:00 pm: watched a movie with his son, Eddy
        10:00 pm: got ready for bed and slept

        Plan: 
        {plan}
        Plan in intervals of {time_interval}:
        """
        
        resulting_plan = None
        attempts = 0
        while not GenerativeAgent.check_plan_format(resulting_plan) and attempts < max_attempts:
            resulting_plan = OpenAILLM.get_llm_response(prompt)
            if DatetimeNL.get_time_nl(curr_time) != "12:00 am":
                resulting_plan = f"{DatetimeNL.get_time_nl(curr_time)}:" + resulting_plan
            resulting_plan = resulting_plan.split('\n')
            resulting_plan = '\n'.join([plan_item for plan_item in resulting_plan if plan_item.strip()])

            attempts += 1
            logging.info(f"planning {time_interval} attempt number {attempts} / {max_attempts}")
            logging.info(resulting_plan)

        if attempts == max_attempts:
            raise ValueError(f"Plan {time_interval} generation failed")

        self.add_to_memory(activity=resulting_plan, curr_time=curr_time, memory_type=f"{time_interval} plan")
        return resulting_plan
    
    

    def get_relevance_scores(self, query):
        query_embedding = OpenAILLM.get_embeddings(query)
        logging.info(json.dumps([memory_item["activity"] for memory_item in self.memory], indent=4))
        memory_item_embeddings = [OpenAILLM.get_embeddings(memory_item["activity"]) for memory_item in self.memory]
        scores = cosine_similarity([query_embedding], memory_item_embeddings)[0]
        return scores

    def calculate_recency_score(self, time0, time1):
        duration_hours = (time1 - time0).total_seconds() // 3600
        score = 0.99**duration_hours
        return score

    def min_max_scaling(self, scores):
        # if min == max, all scores == 1
        min_score = min(scores)
        max_score = max(scores)
        scaled_scores = [(score-min_score+1e-10) / (max_score-min_score+1e-10) for score in scores]
        return scaled_scores

    def combine_scores(self, relevance_scores, importance_scores, recency_scores, relevance_alpha=1, importance_alpha=1, recency_alpha=1):
        combined_scores = []
        for i in range(len(relevance_scores)):
            combined_score = relevance_scores[i] * relevance_alpha
            combined_score += importance_scores[i] * importance_alpha
            combined_score += recency_scores[i] * recency_alpha
            combined_scores.append(combined_score)
        return combined_scores

    def retrieve_memories(self, query, curr_time, top_n=5, timestamp=False):
        relevance_scores = self.get_relevance_scores(query)
        importance_scores = [memory_item["importance"] for memory_item in self.memory]
        recency_scores = [self.calculate_recency_score(memory_item["last_access_time"], curr_time) for memory_item in self.memory]
        combined_scores = self.combine_scores(
            self.min_max_scaling(relevance_scores), 
            self.min_max_scaling(importance_scores), 
            self.min_max_scaling(recency_scores)
        )

        ordered_data = np.argsort(combined_scores)[::-1]
        relevant_memory_indices = ordered_data[:top_n]
        if not timestamp:
            memory_statements = [self.memory[i]['activity'] for i in relevant_memory_indices]
        else:
            memory_statements = [DatetimeNL.get_formatted_date_time(self.memory[i]['creation_time']) + ' ' + self.memory[i]['activity'] for i in relevant_memory_indices]
        return memory_statements
    
    def get_questions_for_reflection(self):
        prompt = ", ".join([memory_item["activity"] for memory_item in self.memory[-100:]])
        prompt += "Given only the information above, what are 3 most salient high-level questions we can answer about the subjects in the statements?"
        questions = OpenAILLM.get_llm_response(prompt, max_tokens=100)
        question_list = [question + "?" for question in questions.split("?")]
        return question_list
    
    def reflect(self, curr_time):
        questions = self.get_questions_for_reflection()
        for question in questions:
            memories = self.retrieve_memories(question, curr_time, top_n=15)
            prompt = f"Statements about {self.name}\n"
            index_str_to_memory = {str(i): memories[i] for i in range(len(memories))}
            for i, memory in enumerate(memories):
                prompt += f"{i}. {memory}\n"
            prompt += "What 5 high-level insights can you infer from the above statements? (example format: insight (because of 1, 5, 3))"
            insights = OpenAILLM.get_llm_response(prompt)
            # remove the 1. 2. or 3. 
            insights_list = [' '.join(insight.strip().split(' ')[1:]) for insight in insights.split(")")][:5]
            for insight in insights_list:
                insight_pair = insight.split("(")
                insight_only, reason = insight_pair
                source_nodes = [node.strip() for node in reason.replace(' ', ",").split(",") if node.strip().isnumeric()]
                source_memories = [index_str_to_memory[source_node] for source_node in source_nodes]
                self.add_to_memory(
                    activity=insight_only.strip(), 
                    curr_time=curr_time,
                    source_memories=source_memories,
                    memory_type="reflect"
                )
        return insights


    @cache
    def calculate_importance(self, memory_statement):
        #example memory statement -  buying groceries at The Willows Market and Pharmacy
        prompt = f'''
        On the scale of 1 to 10, where 1 is purely mundane
        (e.g., brushing teeth, making bed) and 10 is
        extremely poignant (e.g., a break up, college
        acceptance), rate the likely poignancy of the
        following piece of memory.
        Memory: {memory_statement}
        Rating:'''
        return int(OpenAILLM.get_llm_response(prompt, max_tokens=1))

    def get_agent_information(self, aspect="core characteristics", curr_time=None):
        memory_query = f"{self.name}'s {aspect}"
        memory_statements = self.retrieve_memories(memory_query, curr_time)
        joined_memory_statements = '\n- '.join(memory_statements)
        prompt = f"""How would one describe {memory_query} given the following statements?\n- {joined_memory_statements}"""
        return OpenAILLM.get_llm_response(prompt)


    @cache
    def get_agent_summary_description(self, curr_time):
        """
        In our implementation, this summary comprises agents’
        identity information (e.g., name, age, personality), as well as a
        description of their main motivational drivers and statements that
        describes their current occupation and self-assessment.

        This is currently cached using the key curr_time, but can be cached based on the day or hour
        """

        core_characteristics = self.get_agent_information(aspect="core characteristics", curr_time=curr_time)
        current_daily_occupation = self.get_agent_information(aspect="current daily occupation", curr_time=curr_time)
        feelings = self.get_agent_information(aspect="feeling about his recent progress in life", curr_time=curr_time)

        description = f"""
        Name: {self.name} (age: {self.age})
        Innate traits: {', '.join(self.traits)}
        {core_characteristics}
        {current_daily_occupation}
        {feelings}
        """
        return description

    @staticmethod
    def convert_to_emoji(activity):
        prompt = f"Please represent ```{activity}''' using 2 emoji characters"
        return OpenAILLM.get_llm_response(prompt, max_tokens=8)

    def get_curr_location_nodes(self, location):
        location_nodes = []
        while location.contains:
            found = False
            for one_contains in location.contains:
                if self.name in one_contains.agents:
                    location_nodes.append(one_contains)
                    location = one_contains
                    found = True
            if not found:
                raise ValueError(f"Agent {self.name} not in Map")
        return location_nodes

    def get_agent_location(self, activity, curr_time, world_location, max_attempts=5):
        # currently assuming that everyone knows of a set of global locations
        # in paper, the locations that each user knows is based on where they have been to
        
        self_summary = self.get_agent_summary_description(curr_time)
        curr_location_nodes = self.get_curr_location_nodes(world_location)

        curr_location_str = ': '.join([node.name for node in curr_location_nodes])#"Lin family’s house: Eddy Lin’s bedroom: desk"
        curr_location_children = curr_location_nodes[0].get_children_nl() #"Mei and John Lin’s bedroom, Eddy Lin’s bedroom, common room, kitchen, bathroom, garden"
        
        # John is planning to <activity> if activity doesn't start with name
        activity_formatted = activity if activity.startswith(self.name) else f"{self.name} is planning to " + activity

        # next location determined based on general area to explore

        options = world_location.get_children_nl() #The Lin family’s house, Johnson Park, Harvey Oak Supply Store, The Willows Market and Pharmacy, Hobbs Cafe, The Rose and Crown Pub.
        area = []
        attempts = 0
        while options and attempts < max_attempts:
            prompt = f"""
            {self_summary}
            {self.name} is currently in the {curr_location_str} that has {curr_location_children}
            * Prefer to stay in the current area if the activity can be done there.
            {activity_formatted}. Which area should {self.name} go to?
            Choose from one of the options below only:
            {options}
            """
            generated_area_part = OpenAILLM.get_llm_response(prompt)
            matched_area_part = GenerativeAgent.fuzzy_match(generated_area_part, options.split(', '))
            attempts += 1
            # print("Prompt: ", prompt)
            # print("Generated area part: ", generated_area_part)
            # print("Matched area part: ", matched_area_part)
            if matched_area_part is not None:
                options = world_location.name_to_node[matched_area_part].get_children_nl()
                area.append(matched_area_part)
        
        # cannot find new location, use current location instead
        if attempts == max_attempts:
            return curr_location_str.split(': ')
        # move agent on world map
        self.move_agent(world_location, area)
        return area
    
    def move_agent(self, world_location, new_location_nl_list):
        curr_location_nodes = self.get_curr_location_nodes(world_location)
        for curr_node in curr_location_nodes:
            curr_node.remove_agent(self.name)
        for location_nl in new_location_nl_list:
            node = world_location.name_to_node[location_nl]
            node.add_agent(self.name)

    # @staticmethod
    # def get_location(activity):
    #     generated_location = None
    #     while generated_location is None:
    #         prompt = f"Where is ```{activity}''' taking place? Choose only within this list {ALLOWED_LOCATIONS}"
    #         generated_location = OpenAILLM.get_llm_response(prompt)
    #         generated_location = GenerativeAgent.fuzzy_match(generated_location, ALLOWED_LOCATIONS)
    #     return generated_location

    @staticmethod
    def fuzzy_match(generated, allowed_list):
        # can consider semantic search in future but for now, ask it to regenerate if not found
        # exact match

        for allowed_value in allowed_list:
            if generated.lower() == allowed_value.lower():
                return allowed_value
        else:
            # correct value is a substring of generated
            for allowed_value in allowed_list:
                if allowed_value.lower() in generated.lower():
                    return allowed_value
        return None

    @cache
    def get_agent_action_generative(self, curr_time):
        #get what an agent is doing at a specific time given a plan (generated externally first and added to memory)
        formatted_date_time = DatetimeNL.get_formatted_date_time(curr_time)
        query = f"{formatted_date_time}\nIn one sentence (starting with {self.name}), what is {self.name} doing?"
        memories = self.retrieve_memories(query, curr_time, top_n=5)
        joined_memory_statements = '\n- '.join(memories)
        prompt = f"{joined_memory_statements} {query}"
        activity = OpenAILLM.get_llm_response(prompt)
        self.add_to_memory(activity=activity, curr_time=curr_time, memory_type="action")
        return activity

    @staticmethod
    def remove_formatting_before_time(one_string):
        for i, char in enumerate(one_string):
            if char.isdigit():
                return one_string[i:]
        return ''

    def get_agent_action_retrieval_only(self, curr_time):
        # Similar to pGenerative Agent paper, only carries out the plan no new action needed
        plan = [memory_item for memory_item in self.memory if memory_item['memory_type'] == '15 minutes plan'][-1]['activity']
        plan = plan.lower()

        time_nl = DatetimeNL.get_time_nl(curr_time)
        plan_items = plan.split('\n')

        #strip out hypens before time
        plan_items = [GenerativeAgent.remove_formatting_before_time(plan_item) for plan_item in plan_items if GenerativeAgent.remove_formatting_before_time(plan_item)]

        # exact match when the 15 min plan has time for every 15 minute intervals
        for plan_item in plan_items:
            if plan_item.strip().startswith(time_nl):
                return plan_item[len(time_nl)+1:].strip()

        # in case 15 min plan doesn't have all 15 min intervals (e.g. some 30m or 45m intervals are combined together in one entry)
        date = DatetimeNL.get_date_nl(curr_time)

        # if curr_time is later than last entry time, agent is sleep
        last_entry_time_nl = ":".join(plan_items[-1].split(":")[:2])
        if DatetimeNL.convert_nl_datetime_to_datetime(date, last_entry_time_nl) < curr_time:
            return "sleep"
        
        last_activity = None
        for plan_item in plan_items:
            entry_time_nl = ":".join(plan_item.split(":")[:2])
            if DatetimeNL.convert_nl_datetime_to_datetime(date, entry_time_nl) < curr_time:
                last_activity = ':'.join(plan_item.split(":")[2:])
        if last_activity is not None:
            return last_activity
    
        # if not in plan default is sleep
        return "sleep"
    
    def get_plan_after_curr_time(self, curr_time):
        plan = [memory_item for memory_item in self.memory if memory_item['memory_type'] == '15 minutes plan'][-1]['activity']
        time_nl = DatetimeNL.get_time_nl(curr_time)
        plan_items = plan.split('\n')
        #exact match
        for i, plan_item in enumerate(plan_items):
            if plan_item.strip().startswith(time_nl):
                return '\n'.join(plan_items[i:])
        
        #return ''

        # new additions

        date = DatetimeNL.get_date_nl(curr_time)

        last_i_before_curr_time = None
        for i, plan_item in enumerate(plan_items):
            entry_time_nl = ":".join(plan_item.split(":")[:2])
            if DatetimeNL.convert_nl_datetime_to_datetime(date, entry_time_nl) < curr_time:
                last_i_before_curr_time = i
            else:
                break
            
        if last_i_before_curr_time is not None:
            return '\n'.join(plan_items[last_i_before_curr_time:])

        return ''

    @cache
    def get_summary_of_relevant_context(self, other_agent, other_agent_activity, curr_time):
        # here the agent can directly access the other agent's memory which is buggy (feature of generative agent), 
        # maybe can only see a fraction on shared memory (in improved version) based on what they know about the other agent
        prompt1 = f"What is {self.name}’s relationship with the {other_agent.name}?"
        prompt2 = other_agent_activity
        memories1 = self.retrieve_memories(prompt1, curr_time, top_n=5)
        memories2 = other_agent.retrieve_memories(prompt2, curr_time, top_n=5)
        joined_memory_statements = '\n- '.join(memories1 + memories2)
        prompt = f"Summarize this: {joined_memory_statements}"
        return OpenAILLM.get_llm_response(prompt)

    @staticmethod
    def parse_reaction_response(response):
        """
        The first sentence should either contain Yes or No (and maybe some additional words). If yes, the second sentence onwards tells of the actual reaction
        """
        response_parts = response.split(".")
        if "yes" in response_parts[0].lower() and len(response_parts) > 1:
            full_response = '.'.join(response_parts[1:])
            return full_response
        return None

    def get_agent_reaction_about_another_agent(self, other_agent, curr_time):
        #TODO: right now the reaction is only to another agent but in the game world, the agent can respond to other objects as well

        # template is changed to ask for only 1 sentence

        self_activity = self.get_agent_action_retrieval_only(curr_time)
        other_activity = other_agent.get_agent_action_retrieval_only(curr_time)

        formatted_date_time = DatetimeNL.get_formatted_date_time(curr_time)
        summary_of_relevant_context = self.get_summary_of_relevant_context(other_agent, other_activity, curr_time)
        self_summary = self.get_agent_summary_description(curr_time)
        prompt = f"""
        {self_summary}
        {formatted_date_time}
        {self.name}’s status: {self_activity}
        Observation: {self.name} saw {other_agent.name} {other_activity}
        Summary of relevant context from {self.name}’s memory:
        {summary_of_relevant_context}
        Should {self.name} react to the observation? Please respond with either yes or no. If yes, please also then suggest an appropriate reaction in 1 sentence.
        """
        reaction_raw = OpenAILLM.get_llm_response(prompt)
        # print(f"Raw reaction response by {self.name}:",reaction_raw)
        reaction_processed = GenerativeAgent.parse_reaction_response(reaction_raw)
        #based on the paper, need to re-plan with every reaction but don't think super helpful here
        # if reaction_processed is not None:
        #     self.plan(curr_time)
        return reaction_processed

    @staticmethod
    def convert_conversation_in_linearized_representation(conversation_history):
        linearized_conversation_history = 'Here is the dialogue history:\n'
        for turn in conversation_history:
            #sometime conversation last turn has no text
            if turn['text'] is not None:
                linearized_conversation_history += f"{turn['name']}: {turn['text']}\n"
        return linearized_conversation_history

    def speak_to_other_agent(self, other_agent, curr_time, reaction=None, conversation_history=[]):
        if reaction is None:
            return None
        
        self_summary = self.get_agent_summary_description(curr_time)
        formatted_date_time = DatetimeNL.get_formatted_date_time(curr_time)
        self_activity = self.get_agent_action_retrieval_only(curr_time)
        other_activity = other_agent.get_agent_action_retrieval_only(curr_time)
        summary_of_relevant_context = self.get_summary_of_relevant_context(other_agent, other_activity, curr_time)
        
        if not conversation_history:
            # first turn, use only the intent of the speaker to ground
            background = f"{self.name} hopes to do this: {reaction}"
        else:
            # else continue the conversation
            background = GenerativeAgent.convert_conversation_in_linearized_representation(conversation_history)
        
        #What would he say next to {other_agent.name}? Please respond in a conversational style.
        prompt = f"""
        {self_summary}
        {formatted_date_time}
        {self.name}’s status: {self_activity}
        Observation: {self.name} saw {other_agent.name} {other_activity}
        Summary of relevant context from {self.name}’s memory:
        {summary_of_relevant_context}
        {background}
        What would {self.name} say next to {other_agent.name}?
        {self.name}:"""
        return OpenAILLM.get_llm_response(prompt)

    def dialogue(self, other_agent, curr_time, max_turns=10):
        
        #if the reaction indicates an interaction between agents, we generate their dialogue.
        #reaction need not necessarily result in dialogue (can be change in self plan)
        conversation_history = []

        while len(conversation_history) < max_turns and (not conversation_history or conversation_history[-1]["reaction"] is not None):
            # self turn
            response_self = self.get_agent_reaction_about_another_agent(other_agent, curr_time)
            speak_self = self.speak_to_other_agent(other_agent, curr_time, reaction=response_self, conversation_history=conversation_history)
    
            conversation_history.append({
                "name": self.name, 
                "text": speak_self, 
                "reaction": response_self
            })

            if conversation_history[-1]["reaction"] is None:
                return conversation_history

            logging.info(json.dumps(conversation_history[-1], indent=4))
            
            # other turn 
            response_other = other_agent.get_agent_reaction_about_another_agent(self, curr_time)
            speak_other = other_agent.speak_to_other_agent(self, curr_time, reaction=response_other, conversation_history=conversation_history)

            conversation_history.append({
                "name": other_agent.name, 
                "text": speak_other, 
                "reaction": response_other
            })
            logging.info(json.dumps(conversation_history[-1], indent=4))
        linearized_conversation_history = GenerativeAgent.convert_conversation_in_linearized_representation(conversation_history)

        #add dialogue to memory of both agents
        self.add_to_memory(linearized_conversation_history, curr_time, memory_type="dialogue")
        other_agent.add_to_memory(linearized_conversation_history, curr_time, memory_type="dialogue")

        return conversation_history

    def get_status_json(self, curr_time, world_location):
        activity = self.get_agent_action_retrieval_only(curr_time)
        activity_emoji = GenerativeAgent.convert_to_emoji(activity)
        location = self.get_agent_location(activity, curr_time, world_location)
        most_recent_15m_plan = [memory_item for memory_item in self.memory if memory_item['memory_type'] == '15 minutes plan'][-1]['activity']
        status = {
            "name": self.name,
            "activity": activity,
            "activity_emoji": activity_emoji,
            "most_recent_15m_plan": most_recent_15m_plan,
            "location": location, 
        }
        return status