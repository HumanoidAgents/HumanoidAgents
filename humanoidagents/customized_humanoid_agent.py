from humanoid_agent import HumanoidAgent

class CustomizedHumanoidAgent(HumanoidAgent):
    def __init__(name: str, description: str, age: int, traits: list, example_day_plan: str, social_relationships={}, basic_needs=None, emotion=None, **kwargs):
        super().__init__(name, description, age, traits, example_day_plan, social_relationships=social_relationships, basic_needs=basic_needs, emotion=emotion)
        # Add your own initialization of other attributes you would like to add

        
        # if empathy influences the content of conservation with other agents, you only need to override certain methods (e.g. speak_to_other_agent)
        # e.g. self.empathy = 5

        raise NotImplementedError

    def analyze_agent_activity(self, activity):
        super().analyze_agent_activity(activity)

        # Add further ideas on how activities can influence agent states 

        raise NotImplementedError

    @cache
    def get_agent_action_retrieval_based(self, curr_time):
        #Determine how agent states can influence activities
        raise NotImplementedError
    
    
    def get_agent_reaction_about_another_agent(other_agent, curr_time):
        # Determines whether an agent decide whether to starting conversing with another agent

        raise NotImplementedError

    def get_agent_reaction_to_another_agent_utterance_e2e(self, other_agent_utterance, other_agent, curr_time):
        # Determines whether an agent will decide to continue conversing with another agent (after they've started conversing)

        raise NotImplementedError

    def speak_to_other_agent(self, other_agent, curr_time, reaction=None, conversation_history=[]):

        # Determines what an agent says to another agent

        raise NotImplementedError

    def get_sentiment_about_conversation(self, linearized_conversation_history, other_agent):

        # Determines how a conversation will influence the state of an agent (i.e. emotions and closeness to another agent)
        raise NotImplementedError




