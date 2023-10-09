import argparse
import textwrap
import numpy as np
import json
import os
import warnings
from datetime import datetime


import pandas as pd
from dash import Dash, html, dcc, callback, Output, Input
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc

from utils import load_json_file, DatetimeNL

warnings.simplefilter(action='ignore', category=FutureWarning)

def get_filename_list_all(folder_name):
    filename_list = []
    for filename in os.listdir(folder_name):
        if filename.endswith(".json"):
            filename_list.append(os.path.join(folder_name, filename))
    return filename_list

def get_filename_list_by_dates_of_interest(dates_of_interest, folder_name):
    specific_times_of_interest = []
    filename_list = []
    for hour in range(6, 24):
        for minutes in ['00', '15', '30', '45']:
            hour_str = str(hour) if hour > 9 else '0' + str(hour)
            total_time = f"{hour_str}:{minutes}"
            specific_times_of_interest.append(total_time)
    for curr_date in dates_of_interest:
        for specific_time in specific_times_of_interest:
            filename = f"{folder_name}/state_{curr_date}_{specific_time.replace(':','h')}.json"
            filename_list.append(filename)
    return filename_list

def change_str_to_datetime(date, time):
    d_string = f"{date} {time}"
    return datetime.strptime(d_string, '%b %d %Y %I:%M %p')

def get_df_of_agent_activity_basic_status(filename_list): 
    all_data = []
    for filename in filename_list:
        data = load_json_file(filename)
        for agent in data['agents']:
            one_dp = {
                "date": ' '.join(data['date'].split(" ")[1:]),
                "weekday": data['date'].split(" ")[0],
                "time": data['time'],
                "datetime": change_str_to_datetime(' '.join(data['date'].split(" ")[1:]),data['time']),
                "agent_name": agent['name'],
                "activity": agent['activity'],
                "emotion": agent['emotion']
            }
            for key, value in agent['basic_needs'].items():
                one_dp[f"basicneeds_{key}"] = value
            agent_names = [agent['name'] for agent in data['agents']]
            for agent_name in agent_names:
                one_dp[f"closeness_{agent_name}"] = agent["social_relationships"][agent_name]['closeness'] if "social_relationships" in agent and agent_name in agent["social_relationships"] else np.NaN
            all_data.append(one_dp)
    return pd.DataFrame.from_dict(all_data)

def get_df_of_conversation(filename_list): 
    all_data = []
    for filename in filename_list:
        data = load_json_file(filename)
        for conversation_place, conversation_turns in data['conversations'].items():
            agent_involved_names = list(set([one_conversation["name"]for one_conversation in conversation_turns[0]]))
            one_dp = {
                "date": ' '.join(data['date'].split(" ")[1:]),
                "weekday": data['date'].split(" ")[0],
                "time": data['time'],
                "datetime": change_str_to_datetime(' '.join(data['date'].split(" ")[1:]),data['time']),
                "conversation_place": conversation_place,
                "conversation_turns": conversation_turns[0],
                "agent_involved_names": agent_involved_names
            }
            all_data.append(one_dp)
    return pd.DataFrame.from_dict(all_data)

def rename_column_names_for_legend(col):
    if col.startswith("basicneed") or col.startswith("closeness"):
        return col.split("_")[-1]
    return col

@callback(
    Output('basic-status-graph', 'figure'),
    Input('agent-selection', 'value'),
    Input('tooltip-selection', 'value')
)
def update_graph(agent_value, tooltip_value):
    df_selected_agent = df_agent_activity_basic_status[df_agent_activity_basic_status.agent_name==agent_value]
    basic_needs_columns = [col for col in df_selected_agent.columns if col.startswith('basicneeds')]
    basic_status_columns = ['datetime'] + basic_needs_columns
    df_basic_status = df_selected_agent[basic_status_columns]
    df_basic_status.columns = [rename_column_names_for_legend(col) for col in basic_status_columns]
    
    if tooltip_value == 'one basic need value with activity detail and emotion':
        fig = px.line(
        df_basic_status,
        x='datetime',
        y = [c for c in df_basic_status.columns if c != "datetime"],

        custom_data=[df_selected_agent["activity"].apply(lambda txt: '<br>'.join(textwrap.wrap(txt, width=50))),
                     df_selected_agent["emotion"],
                     df_selected_agent["activity"]],
        markers=True
        )
        fig.update_xaxes(showspikes=True)
        fig.update_traces(mode="markers+lines",hovertemplate="<b>Time:</b> %{x} <br><b>Value:</b> %{y}  <br><b>Activity:</b> %{customdata[0]}<br><b>Emotion:</b> %{customdata[1]}")
        fig.update_layout(legend_title="Basic Needs")
    elif tooltip_value == 'all basic needs values':
        fig = px.line(
        df_basic_status,
        x='datetime',
        y = [c for c in df_basic_status.columns if c != "datetime"],
        custom_data=[df_selected_agent["activity"].apply(lambda txt: '<br>'.join(textwrap.wrap(txt, width=50))),
                df_selected_agent["emotion"],
                df_selected_agent["activity"]],
        markers=True
        )
        fig.update_traces(mode="markers+lines",hovertemplate="%{y}")
        fig.update_layout(hovermode='x unified',
                          legend_title="Basic Needs",
                          hoverlabel=dict(bgcolor='rgba(255,255,255,0.75)',
                                          font=dict(color='black')))

    return fig

@callback(
    Output('hover-data', 'children'),
    Input('basic-status-graph', 'hoverData'))
def display_activity_and_emotion_data(hoverData):
    if hoverData != None:
        activity_data = '\n'.join(textwrap.wrap(json.dumps(hoverData['points'][0]['customdata'][2]).strip('\"').strip(' '), width=80))
        emotion_data = json.dumps(hoverData['points'][0]['customdata'][1]).strip('\"').strip(' ')
        return f'*Activity*: {activity_data}\n\n*Emotion*: {emotion_data}'
    else:
        return hoverData
    
@callback(
    Output('closeness-graph', 'figure'),
    Input('agent-selection', 'value')
)
def update_graph(value):
    df_selected_agent = df_agent_activity_basic_status[df_agent_activity_basic_status.agent_name==value]
    col_names_social_relationship = [column_name for column_name in df_agent_activity_basic_status.columns if column_name.startswith('closeness') == True]
    col_names_social_relationship_without_self = [column_name for column_name in col_names_social_relationship if column_name.split("_")[-1] != value]
    columns_selected = col_names_social_relationship_without_self + ['datetime']
    df_social_relationship_selected = df_selected_agent[columns_selected]
    df_conversations_selected_agent = df_conversations[df_conversations.apply(lambda x: value in list(x['agent_involved_names']), axis=1)]
    if df_conversations_selected_agent.empty:
        df_conversations_selected_agent = pd.DataFrame(columns=['date','weekday','time','datetime','conversation_place','conversation_turns','agent_involved_names'])
    df_conversations_selected_agent['datetime'] = pd.to_datetime(df_conversations_selected_agent['datetime'])

    df_new = pd.merge(df_selected_agent, df_conversations_selected_agent, on=['date','weekday','time','datetime'], how='left')
    
    all_columns = df_new.columns
    df_new.columns = [rename_column_names_for_legend(col) for col in all_columns]
    df_social_relationship_selected_columns = df_social_relationship_selected.columns
    df_social_relationship_selected.columns = [rename_column_names_for_legend(col) for col in df_social_relationship_selected_columns]
    fig_closeness = px.line(
    df_new,
    x='datetime',
    y = [c for c in df_social_relationship_selected.columns if c != "datetime"],
    custom_data=[df_new['conversation_place'],\
                 df_new['agent_involved_names'],\
                 df_new['conversation_turns']
                ],
    markers=True
    )
    fig_closeness.update_traces(mode="markers+lines",hovertemplate="%{y}")
    fig_closeness.update_layout(hovermode='x unified',
                                legend_title="Closeness",
                        hoverlabel=dict(bgcolor='rgba(255,255,255,0.75)',
                                        font=dict(color='black')))
    return fig_closeness


@callback(
    Output('hover-conversation-data', 'children'),
    Input('closeness-graph', 'hoverData'))
def display_conversation_data(hoverData):
    if hoverData != None:
        hoverData_dict = json.loads(json.dumps(hoverData))
        conversation_place_data = json.dumps(hoverData['points'][0]['customdata'][0]).strip('\"').strip(' ')
        agent_involved_data = json.dumps(hoverData['points'][0]['customdata'][1]).strip('\"').strip(' ')
        conversation_turns_data = hoverData_dict['points'][0]['customdata'][2]
        conversation_output_string = ''
        if conversation_turns_data != None:
            for conversation in conversation_turns_data:
                conversation_output_string += f'\n-----------------------'
                if conversation != None:
                    for key, value in conversation.items():
                        if type(value) != str:
                            value = str(value)
                        value_str = '\n'.join(textwrap.wrap(value.strip('\"').strip(' '), width=80))
                        conversation_output_string += f'\n{key}:\n{value_str}\n'
                else:
                    conversation_output_string = ' None'
        else:
            conversation_output_string = ' None'
        return f'*Place*: {conversation_place_data}\n\n*Agents involved*: {agent_involved_data}\n\n*Conversations*:{conversation_output_string}'
    else:
        return hoverData

def get_layout():
    layout = html.Div(className='column', children=[
                html.Div([
                            html.Br(),
                            html.H1(children='Realtime dashboard for humanoid agent', style={'textAlign':'center'}),
                            html.P("Select agent:"),
                            dcc.Dropdown(df_agent_activity_basic_status.agent_name.unique(), id='agent-selection'),
                            html.Br(),
                            html.H2(children='Basic Need Graph of Selected Agent', style={'textAlign':'center'}),
                            html.P("Select what to show in tooltip:"),
                            dbc.RadioItems(
                                id='tooltip-selection',
                                inline=True,
                                options=['all basic needs values', 'one basic need value with activity detail and emotion'],
                                value='one basic need value with activity detail and emotion'
                            ),
                            dcc.Graph(id='basic-status-graph'),
                            html.Div([
                                        dcc.Markdown(""" **Activity and Emotion Data** (Mouse over values in the graph.) """),
                                        html.Pre(id='hover-data')
                                    ]),
                        ]),
                html.Div([
                        html.H2(children='Social Relationship: Closeness with Other Agents', style={'textAlign':'center'}),
                        dcc.Graph(id='closeness-graph'),
                        html.Div([
                                    dcc.Markdown(""" **Conversations Data** (Mouse over values in the graph.) """),
                                    html.Pre(id='hover-conversation-data')
                                ]),
                        ])
                    
                ], style={'padding': '0px 20px 20px 20px'})
    return layout


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'HumanoidAgents-VisualizationDashboard',
                                     description = 'Visualizing the humanoid agents attributes by Python Dash')
    parser.add_argument('-f', '--folder', required=True, help='folder containing generations from run_simulation.py')
    parser.add_argument('-m', '--mode',choices=['all','date_range'], default="all")
    parser.add_argument('-s', '--start_date', help='Enter start date (inclusive) by YYYY-MM-DD e.g.2023-01-03', default="2023-01-03")
    parser.add_argument('-e', '--end_date', help='Enter end date (inclusive) by YYYY-MM-DD e.g.2023-01-04', default="2023-01-03")

    args = parser.parse_args()

    folder_name = args.folder
    mode = args.mode
    start_date = args.start_date
    end_date = args.end_date
    filename_list = get_filename_list_all(folder_name)


    if mode == 'all':
        filename_list = get_filename_list_all(folder_name)
    elif mode == 'date_range':
        dates_of_interest = DatetimeNL.get_date_range(start_date, end_date)
        filename_list = get_filename_list_by_dates_of_interest(dates_of_interest, folder_name)
    df_agent_activity_basic_status = get_df_of_agent_activity_basic_status(filename_list).sort_values(by=['datetime'])
    df_conversations = get_df_of_conversation(filename_list)
    if df_conversations.empty == False:
        df_conversations = df_conversations.sort_values(by=['datetime'])

    external_scripts = []
    external_stylesheets = [dbc.themes.LUMEN]

    app = Dash(__name__, 
               title='Humanoid Agent Analytics', 
               update_title='Loading...',
               external_scripts=external_scripts,
               external_stylesheets=external_stylesheets)

    app.layout = get_layout()
    app.run(debug=True)
