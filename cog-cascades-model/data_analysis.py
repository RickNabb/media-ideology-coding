from enum import Enum
from random import *
from utils import *
from statistics import mean, variance, mode
from copy import deepcopy
from plotting import *
from nlogo_colors import *
import itertools
import pandas as pd
import os
from os.path import exists
import numpy as np
from scipy.stats import chi2_contingency, truncnorm, pearsonr
from sklearn.linear_model import LinearRegression
import math
import matplotlib.pyplot as plt
from nlogo_io import *
from messaging import dist_to_agent_brain, believe_message
# import statsmodels.formula.api as smf

DATA_DIR = 'D:/school/grad-school/Tufts/research/gallup-media-mask/simulation-data'

"""
STATS STUFF
"""

'''
Return a tuple of summary statistical measures given a list of values.

:param l: The list to calculate stats for.
'''
def summary_statistics(l):
  if len(l) >= 2:
    return (mean(l), variance(l), mode(l))
  elif len(l) >= 1:
    return (mean(l), -1, mode(l))
  else:
    return (-1, -1, -1)

"""
Sample a distribution given a specific attribute. This distribution may
also depend on another, and if so, the function recursively calls
itself to return the needed dependency.

:param attr: An attribute from the Attribues enumeration to sample
its approriate distribution in the empirical data.
"""
def random_dist_sample(attr, resolution, given=None):
    return AttributeValues[attr.name]['vals'](resolution)

"""
Sample an attribute with an equal distribution over the values.

:param attr: The attribute to sample - e.g. Attributes.I.
"""
def random_sample(attr):
  rand = int(math.floor(random() * len(list(attr.value))))
  val = list(attr.value)[rand]
  return val.value

def test_random_sample():
  print(random_sample(Attributes.VG))

"""
ANALYSIS FUNCTIONS
"""

def process_multiple_sim_data(path):
  for file in os.listdir(path):
    data = process_sim_data(f'{path}/{file}')
    stats = citizen_message_statistics(data[0], data[1])

'''
Parse a NetLogo chart export .csv file. This requires a single chart export file
and should not be run on an entire world export. This will return a dictionary
of plot points in a data frame, keyed by the pen name.

:param path: The path to the chart .csv file.
'''
def process_chart_data(path):
  f = open(path)
  raw = f.read()
  f.close()
  chunks = raw.split('\n\n')

  model_lines = chunks[1].replace('"','').split('\n')
  model_keys = model_lines[1].split(',')
  model_vals = model_lines[2].split(',')
  model_props = { model_keys[i]: model_vals[i] for i in range(len(model_keys)) }

  prop_lines = chunks[2].replace('"','').split('\n')
  chart_props = {}
  chart_props['color'] = {}
  keys = prop_lines[1].split(',')
  vals = prop_lines[2].split(',')
  for i in range(0, len(keys)):
    chart_props[keys[i]] = vals[i]

  data_sets = {}
  chart_lines = chunks[4].split('\n')
  
  data_set_splits = []
  split_line = chart_lines[0].split(',')
  for i in range(0, len(split_line)):
    el = split_line[i].replace('"','')
    if el != '':
      data_set_splits.append((el, i))
  for split in data_set_splits:
    data_sets[split[0]] = []

  for i in range(1, len(chart_lines)):
    line = chart_lines[i].replace('"','')
    if line == '': continue

    els = line.split(',')
    for j in range(0, len(data_set_splits)):
      split = data_set_splits[j]
      if j+1 == len(data_set_splits):
        data_sets[split[0]].append(els[split[1]:])
      else:
        data_sets[split[0]].append(els[split[1]:data_set_splits[j+1][1]])

  dfs = {}
  for split in data_set_splits:
    # df = pd.DataFrame(data=data_sets[split[0]][1:], columns=data_sets[split[0]][0])
    df = pd.DataFrame(data=data_sets[split[0]][1:], columns=data_sets[split[0]][0])
    del df['pen down?']
    chart_props['color'][split[0]] = df['color'].iloc[0] if len(df['color']) > 0 else 0
    del df['color']
    # del df['x']
    dfs[split[0]] = df

  return (model_props, chart_props, dfs)

'''
Read multiple NetLogo chart export files and plot all of them on a single
Matplotlib plot.

:param in_path: The directory to search for files in.
:param in_filename: A piece of filename that indicates which files to parse
in the process. This should usually be the name of the chart in the NetLogo file.
'''
def process_multi_chart_data(in_path, in_filename='percent-agent-beliefs'):
  props = []
  multi_data = []
  model_params = {}
  print(f'process_multi_chart_data for {in_path}/{in_filename}')

  if not os.path.isdir(in_path):
    print(f'ERROR: Path not found {in_path}')
    return (-1, -1, -1)
  if len(os.listdir(in_path)) == 0:
    print(f'ERROR: Path contains no files')
    return (-1,-1,-1)

  for file in os.listdir(in_path):
    if in_filename in file and '.swp' not in file and '.swo' not in file:
      data = process_chart_data(f'{in_path}/{file}')
      model_params = data[0]
      props.append(data[1])
      multi_data.append(data[2])

  full_data_size = int(model_params['tick-end']) + 1
  means = { key: [] for key in multi_data[0].keys() }
  for data in multi_data:
    for key in data.keys():
      data_vector = np.array(data[key]['y']).astype('float32')

      if len(data_vector) != full_data_size:
        if len(data_vector) > full_data_size:
          data_vector = data_vector[:full_data_size]
        # TODO: This is janky code
        # elif abs(len(data_vector) - full_data_size) <= 5:
        #   data_vector = np.append(data_vector, [ data_vector[-1] for i in range(abs(len(data_vector) - full_data_size)) ])
        else:
          print(f'ERROR parsing multi chart data for pen "{key}" -- data length {len(data_vector)} did not equal number of ticks {full_data_size}')
          continue

      if means[key] == []:
        means[key] = np.array([data_vector])
      else:
        means[key] = np.vstack([means[key], data_vector])

  final_props = props[0]
  props_y_max = np.array([ float(prop['y max']) for prop in props ])
  final_props['y max'] = props_y_max.max()
  return (means, final_props, model_params)

def process_message_data(in_path, rand_id):
  '''
  Process message data from the simulation runs: messages heard by agents,
  believed by agents, agent belief data over time, and messages sent by
  institutional agents.

  As of now, the analysis calculates a timeseries of the mean differences
  between agent beliefs at each tick and the messages they hear, and believe
  (two separate timeseries).

  :param in_path: The path to a single param combo run's output
  :param rand_id: The random seed used for filenames to process.

  Analysis ideas:
  [X] Mean and var difference of message exposure (heard) over time
  [X] Mean and var difference of message belief over time
  - Change in media messaging over time (color chart over time -- maybe stack plot)
  - Do per run and then aggregate across runs with same paratmeres
  '''
  print(f'processing message data for {in_path}/{rand_id}_FILE.json')
  citizen_beliefs_file = open(f'{in_path}/{rand_id}_bel_over_time.json', 'r')
  messages_believed_file = open(f'{in_path}/{rand_id}_messages_believed.json', 'r')
  messages_heard_file = open(f'{in_path}/{rand_id}_messages_heard.json', 'r')
  messages_sent_file = open(f'{in_path}/{rand_id}_messages_sent.json', 'r')

  '''
  Format of citizen_beliefs: [ {'tick': [{'A': 1}}, {'1': {'A': 2}] } ... where each key is a tick, and each array entry in a tick is a citizen ]

  Format of messages_believed: [ [{'tick': [believed], 'tick': [believed], ...}] where each array is a citizen ]
  '''

  belief_data = json.load(citizen_beliefs_file)
  bel_data = {}
  for tick in belief_data:
    bel_data.update(tick)
  belief_data = bel_data

  messages_heard_data = json.load(messages_heard_file)
  messages_heards_data = []
  for cit in messages_heard_data:
    messages_heard = {}
    for entry in cit:
      messages_heard.update(entry)
    messages_heards_data.append(messages_heard)
  messages_heard_data = messages_heards_data

  messages_bel_data = json.load(messages_believed_file)
  messages_belief_data = []
  for cit in messages_bel_data:
    messages_believed = {}
    for entry in cit:
      messages_believed.update(entry)
    messages_belief_data.append(messages_believed)
  messages_bel_data = messages_belief_data

  messages_sent_data = json.load(messages_sent_file)
  all_messages = { }
  for media_messages in messages_sent_data:
    all_messages.update(media_messages)

  citizen_beliefs_file.close()
  messages_believed_file.close()
  messages_heard_file.close()
  messages_sent_file.close()

  ticks = len(belief_data)
  belief_diffs = []
  heard_diffs = []
  for tick in range(ticks):
    belief_diffs_at_tick = np.array([])
    heard_diffs_at_tick = np.array([])
    # print(belief_data[tick])
    belief_data_for_tick = belief_data[str(tick)]
    # heard_data_for_tick = messages_heard_data[tick]
    for cit_id in range(len(belief_data_for_tick)):
      agent_brain = belief_data_for_tick[cit_id]
      agent_malleables = [ bel for bel in agent_brain.keys() ]
      agent_brain['malleable'] = agent_malleables
      messages_believed = []
      messages_heard = []
      
      if str(tick) in messages_bel_data[cit_id].keys():
        messages_believed = messages_bel_data[cit_id][str(tick)]
      if str(tick) in messages_heard_data[cit_id].keys():
        messages_heard = messages_heard_data[cit_id][str(tick)]

      cur_agent_belief = agent_brain
      for message_id in sorted(messages_believed + messages_heard):
        message = all_messages[f'{message_id}']
        diff = dist_to_agent_brain(cur_agent_belief,message) 
        if message_id in messages_heard:
          heard_diffs_at_tick = np.append(heard_diffs_at_tick, diff)
        if message_id in messages_believed:
          belief_diffs_at_tick = np.append(belief_diffs_at_tick, diff)
          cur_agent_belief = believe_message(agent_brain, message, '', 'discrete')
    belief_diffs.append((belief_diffs_at_tick.mean(), belief_diffs_at_tick.var()))
    heard_diffs.append((heard_diffs_at_tick.mean(), heard_diffs_at_tick.var()))
  return { 'believed': np.array(belief_diffs), 'heard': np.array(heard_diffs) }

def process_multi_message_data(in_path):
  '''
  Aggregate all the message-related data analysis for a given experiment
  output path.

  :param in_path: The directory path to a given experiment directory
  containing messaging data files.
  '''
  proxy_filename = 'world.csv'
  file_ids = []
  message_multi_data = []
  if os.path.isdir(in_path):
    for file in os.listdir(in_path):
      if proxy_filename in file:
        file_ids.append(file.replace('_world.csv',''))

    for file_id in file_ids:
      data = process_message_data(in_path, file_id)
      message_multi_data.append(data)
    return message_multi_data
  else:
    print(f'ERROR: Path not found {in_path}')
    return -1

def plot_multi_message_data(multi_data_entry, out_path, show_plot=False):
  line_color = lambda key: '#000000'

  multi_data_has_multiple = lambda multi_data_entry: type(multi_data_entry[0]) == type(np.array(0)) and len(multi_data_entry) > 1

  measures = ['believed','heard']
  param_combo = multi_data_entry[0]
  multi_data = multi_data_entry[1]

  # The case where a path was not found
  if multi_data == -1:
    print(f'ERROR: No data for entry {param_combo}')
    return

  for measure in measures:
    # Check to make sure data can be stacked
    data_lengths = []
    for repetition in multi_data:
      data_lengths.append(len(repetition[measure]))
    if len(set(data_lengths)) > 1:
      print(f'ERROR: Data lengths unequal between repetitions for param combo {param_combo} measure {measure}: {data_lengths}')
      continue

    fig, (ax) = plt.subplots(1, figsize=(8,6))
    y_min = 0
    y_max = 6
    x_min = 0
    x_max = len(multi_data[0]['believed'])
    ax.set_ylim([y_min, y_max])
    plt.yticks(np.arange(y_min, y_max, step=1))
    plt.xticks(np.arange(x_min, x_max*1.1, step=5))
    ax.set_ylabel("Mean distance")
    ax.set_xlabel("Time step")

    mean_vec = np.array([ [ val[0] for val in repetition[measure] ] for repetition in multi_data ])
    mean_vec = mean_vec.mean(0) if multi_data_has_multiple(multi_data) else mean_vec[0]

    var_vec = np.array([ [ val[1] for val in repetition[measure] ] for repetition in multi_data ])
    var_vec = var_vec.var(0) if multi_data_has_multiple(multi_data) else var_vec[0]

    ax.plot(mean_vec, c=line_color(param_combo))
    ax.fill_between(range(x_min, len(mean_vec)), mean_vec-var_vec, mean_vec+var_vec, facecolor=f'{line_color(param_combo)}44')

    plt.savefig(f'{out_path}/{"-".join(param_combo)}_messages_{measure}_dist.png')
    if show_plot: plt.show()
    plt.close()

'''
Given some multi-chart data, plot it and save the plot.

:param multi_data: Data with means and std deviations for each point.
:param props: Properties object for the plotting.
:param out_path: A path to save the results in.
:param out_filename: A filename to save results as, defaults to 'aggregate-chart'
:param show_plot: Whether or not to display the plot before saving.
'''
def plot_multi_chart_data(types, multi_data, props, out_path, out_filename='aggregate-chart', show_plot=False):
  if PLOT_TYPES.LINE in types:
    plot = plot_nlogo_multi_chart_line(props, multi_data)
    plt.savefig(f'{out_path}/{out_filename}_line.png')
    if show_plot: plt.show()
    plt.close()

  if PLOT_TYPES.STACK in types:
    plot = plot_nlogo_multi_chart_stacked(props, multi_data)
    plt.savefig(f'{out_path}/{out_filename}_stacked.png')
    if show_plot: plt.show()
    plt.close()

  if PLOT_TYPES.HISTOGRAM in types:
    plot = plot_nlogo_multi_chart_histogram(props, multi_data)
    plt.savefig(f'{out_path}/{out_filename}_histogram.png')
    if show_plot: plt.show()
    plt.close()

'''
Plot multiple NetLogo chart data sets on a single plot. 

:param props: The properties dictionary read in from reading the chart file. This
describes pen colors, x and y min and max, etc.
:param multi_data: A dictionary (keyed by line) of matrices where each row is one simulation's worth of data points.
'''
def plot_nlogo_multi_chart_stacked(props, multi_data):
  init_dist_width = 10

  # series = pd.Series(data)
  fig, (ax) = plt.subplots(1, figsize=(8,6))
  # ax, ax2 = fig.add_subplot(2)
  ax.set_ylim([0, 1])
  y_min = int(round(float(props['y min'])))
  y_max = int(round(float(props['y max'])))
  x_min = int(round(float(props['x min'])))
  x_max = int(round(float(props['x max'])))
  plt.yticks(np.arange(y_min, y_max+0.2, step=0.2))
  plt.xticks(np.arange(x_min, x_max+10, step=10))
  ax.set_ylabel("Portion of agents who believe b")
  ax.set_xlabel("Time Step")


  multi_data_keys_int = list(map(lambda el: int(el), multi_data.keys()))

  # To use Netlogo colors
  # line_color = lambda key: f"#{rgb_to_hex(NLOGO_COLORS[int(round(float(props['color'][key])))])}"

  # To use higher resolution colors
  resolution = int(max(multi_data_keys_int))+1
  line_color = lambda key: f"#{rgb_to_hex([ 255 - round((255/(resolution-1))*int(key)), 0, round((255/(resolution-1)) * int(key)) ])}"
  
  mean_vecs = []
  var_vecs = []
  rev_keys_int = sorted(multi_data_keys_int, reverse=True)
  rev_keys = list(map(lambda el: f'{el}', rev_keys_int))
  multi_data_has_multiple = lambda multi_data_entry: type(multi_data_entry[0]) == type(np.array(0)) and len(multi_data_entry) > 1
  for key in rev_keys:
    mean_vec = multi_data[key].mean(0) if multi_data_has_multiple(multi_data[key])  else multi_data[key]
    var_vec = multi_data[key].var(0) if multi_data_has_multiple(multi_data[key]) else np.zeros(len(mean_vec))

    # Add padding for the initial values so those are displayed in the graph
    mean_vec = np.insert(mean_vec, 0, [ mean_vec[0] for i in range(init_dist_width) ])

    mean_vecs.append(mean_vec)
    var_vecs.append(var_vec)
  
  ax.set_xlim([x_min-init_dist_width,len(mean_vecs[0])-init_dist_width])
  plt.stackplot(range(x_min-init_dist_width, len(mean_vecs[0])-init_dist_width), mean_vecs, colors=[ f'{line_color(c)}' for c in rev_keys ], labels=[ f'b = {b}' for b in rev_keys ])

'''
Plot multiple NetLogo chart data sets on a single plot. This will scatterplot
each data set and then draw a line of the means at each point through the
entire figure.

:param props: The properties dictionary read in from reading the chart file. This
describes pen colors, x and y min and max, etc.
:param multi_data: A list of dataframes that contain chart data.
'''
def plot_nlogo_multi_chart_line(props, multi_data):
  # series = pd.Series(data)
  # print(multi_data)
  fig, (ax) = plt.subplots(1, figsize=(8,6))
  # ax, ax2 = fig.add_subplot(2)
  y_min = int(round(float(props['y min'])))
  y_max = int(round(float(props['y max'])))
  x_min = int(round(float(props['x min'])))
  x_max = int(round(float(props['x max'])))
  ax.set_ylim([0, y_max])
  plt.yticks(np.arange(y_min, y_max, step=1))
  # plt.yticks(np.arange(y_min, y_max*1.1, step=y_max/10))
  plt.xticks(np.arange(x_min, x_max*1.1, step=5))
  ax.set_ylabel("% of agents who believe b")
  ax.set_xlabel("Time Step")

  line_color = lambda key: '#000000'
  line_names_to_color = {
    'dem': '#0000ff',
    'mod': '#ff00ff',
    'rep': '#ff0000'
  }

  if 'dem' in list(multi_data.keys()):
    line_color = lambda key: line_names_to_color[key]
  elif list(multi_data.keys())[0] != 'default':
    # This is specific code to set the colors for belief resolutions
    multi_data_keys_int = list(map(lambda el: int(el), multi_data.keys()))
    resolution = int(max(multi_data_keys_int))+1
    line_color = lambda key: f"#{rgb_to_hex([ 255 - round((255/max(resolution-1,1))*int(key)), 0, round((255/max(resolution-1,1)) * int(key)) ])}"
 
  multi_data_has_multiple = lambda multi_data_entry: type(multi_data_entry[0]) == type(np.array(0)) and len(multi_data_entry) > 1
 
  for key in multi_data:
    mean_vec = multi_data[key].mean(0) if multi_data_has_multiple(multi_data[key]) else multi_data[key][0]
    var_vec = multi_data[key].var(0) if multi_data_has_multiple(multi_data[key]) else np.zeros(len(mean_vec))
    # print(multi_data[key])
    # print(var_vec)
    ax.plot(mean_vec, c=line_color(key))
    ax.fill_between(range(x_min, len(mean_vec)), mean_vec-var_vec, mean_vec+var_vec, facecolor=f'{line_color(key)}44')
  
  return multi_data

'''
Plot multiple NetLogo chart data sets on a single plot. This will scatterplot
each data set and then draw a line of the means at each point through the
entire figure.

:param props: The properties dictionary read in from reading the chart file. This
describes pen colors, x and y min and max, etc.
:param multi_data: A list of dataframes that contain chart data.
'''
def plot_nlogo_histogram(props, multi_data):
  # series = pd.Series(data)
  # print(multi_data)
  fig, (ax) = plt.subplots(1, figsize=(8,6))
  # ax, ax2 = fig.add_subplot(2)
  ax.set_ylim([0, 1.1])
  y_min = int(round(float(props['y min'])))
  y_max = int(round(float(props['y max'])))
  x_min = int(round(float(props['x min'])))
  x_max = int(round(float(props['x max'])))
  plt.yticks(np.arange(y_min, y_max+0.2, step=0.2))
  plt.xticks(np.arange(x_min, x_max+10, step=10))
  ax.set_ylabel("# of agents who believe b")
  ax.set_xlabel("Time Step")

  line_color = lambda key: '#000000'

  if list(multi_data.keys())[0] != 'default':
    # This is specific code to set the colors for belief resolutions
    multi_data_keys_int = list(map(lambda el: int(el), multi_data.keys()))
    resolution = int(max(multi_data_keys_int))+1
    bar_color = lambda key: f"#{rgb_to_hex([ 255 - round((255/max(resolution-1,1))*int(key)), 0, round((255/max(resolution-1,1)) * int(key)) ])}"

  multi_data_has_multiple = lambda multi_data_entry: type(multi_data_entry[0]) == type(np.array(0)) and len(multi_data_entry) > 1
 
  for key in multi_data:
    mean_vec = multi_data[key].mean(0) if multi_data_has_multiple(multi_data[key])  else multi_data[key]
    var_vec = multi_data[key].var(0) if multi_data_has_multiple(multi_data[key]) else np.zeros(len(mean_vec))
    # print(var_vec)
    ax.plot(mean_vec, c=bar_color(key))
    ax.fill_between(range(x_min, len(mean_vec)), mean_vec-var_vec, mean_vec+var_vec, facecolor=f'{bar_color(key)}44')
  
  return multi_data
      
'''
From a NetLogo world export file, read in the simulation data for citizens
and media entities. They are stored in Pandas dataframes for further processing.

:param path: The path to the file to read data in from.
'''
def process_sim_data(path):
  f = open(path)
  raw = f.read()
  f.close()
  lines = raw.split('\n')
  turtle_data = []
  for i in range(0, len(lines)):
    line = lines[i]
    if line == '"TURTLES"':
      while line.strip() != '':
        i += 1
        line = lines[i]
        turtle_data.append(line.replace('""','"').split(','))

  turtle_data[0] = list(map(lambda el: el.replace('"',''), turtle_data[0]))
  turtle_df = pd.DataFrame(data=turtle_data[1:], columns=turtle_data[0])

  unneeded_cols = ['color', 'heading', 'xcor', 'ycor', 'label', 'label-color', 'shape', 'pen-size', 'pen-mode', 'size','hidden?']
  citizen_delete = ['media-attrs','messages-sent']
  media_delete = ['messages-heard','brain','messages-believed']

  for col in unneeded_cols:
    del turtle_df[col]

  citizen_df = turtle_df[turtle_df['breed'] == '"{breed citizens}"']
  media_df = turtle_df[turtle_df['breed'] == '"{breed medias}"']

  for col in citizen_delete:
    del citizen_df[col]
  for col in media_delete:
    del media_df[col]

  return (citizen_df, media_df)

'''
Get a relevant set of statistics about the citizens' ending state
after the simulation was run: which messages they heard

:param citizen_df: A dataframe containing citizen data.
'''
def citizen_message_statistics(citizen_df, media_df):
  messages = {}
  # Generate a data frame for media messages
  for m in media_df.iterrows():
    m_sent = nlogo_mixed_list_to_dict(m[1]['messages-sent'])
    messages.update(m_sent)
  for m_id, val in messages.items():
    val['id'] = int(m_id.strip())

  message_vals = list(messages.values())
  messages_df = pd.DataFrame(data=message_vals, columns=list(message_vals[0].keys()))

  # Generate citizen data frames relevant for statistics
  heard_dfs = {}
  for citizen in citizen_df.iterrows():
    parsed = nlogo_mixed_list_to_dict(citizen[1]['messages-heard'])
    flat_heard = []
    for timestep,message_ids in parsed.items():
      flat_heard.extend([ { 'tick': int(timestep), 'message_id': m_id  } for m_id in message_ids ] )
    df = pd.DataFrame(flat_heard)
    heard_dfs[int(citizen[1]['who'].replace('"',''))] = df
  
  believed_dfs = {}
  for citizen in citizen_df.iterrows():
    parsed = nlogo_mixed_list_to_dict(citizen[1]['messages-believed'])
    flat_believed = []
    if not type(parsed) == list:
      for timestep,message_ids in parsed.items():
        flat_believed.extend([ { 'tick': int(timestep), 'message_id': m_id  } for m_id in message_ids ] )
      df = pd.DataFrame(flat_believed)
      believed_dfs[int(citizen[1]['who'].replace('"',''))] = df
    else:
      believed_dfs[int(citizen[1]['who'].replace('"',''))] = pd.DataFrame()
  
  # Analyze the data frames for some statistical measures (per citizen)
  # - Total heard
  # - Total believed
  # - Ratio of believed/heard
  # - Totals heard broken down by partisanship & ideology
  # - Totals believed broken down by partisanship & ideology
  # - Totals heard broken down by virus belief
  # - Totals believed broken down by virus belief
  # - Somehow get at beliefs over time?

  per_cit_stats = {}
  for row in citizen_df.iterrows():
    citizen = row[1]
    cit_id = int(citizen['who'].replace('"',''))
    per_cit_stats[cit_id] = per_citizen_stats(cit_id, messages_df, heard_dfs[cit_id], believed_dfs[cit_id])

  # Analyze some group-level measures
  # - Aggregate by citizen's partisan/ideology pair
  # - 

  aggregate_stats = citizens_stats(citizen_df, per_cit_stats)

  return (messages_df, heard_dfs, believed_dfs, per_cit_stats, aggregate_stats)

'''
Generate some statistical measures based on the aggregate view of the citizenry.

:param cit_df: A dataframe containing data for each citizen.
:param per_cit_stats: A dictionary of statistical measures calculated
for each citizen. This is generated from the `citizen_stats()` function.
'''
def citizens_stats(cit_df, per_cit_stats):
  partisanships = list(attrs_as_array(Attributes.P))
  ideologies = list(attrs_as_array(Attributes.I))
  virus_believe_vals = [ -1, 0, 1 ]

  pi_keyed_dict = { (prod[0], prod[1]): 0 for prod in itertools.product(partisanships, ideologies) }
  virus_belief_keyed_dict = { (prod[0], prod[1], prod[2]): 0 for prod in itertools.product(virus_believe_vals, repeat=3) }

  total_by_p = { p: 0 for p in partisanships }
  total_by_i = { i: 0 for i in ideologies }
  total_by_p_i = pi_keyed_dict.copy()

  heard_by_cit_p_i = { (prod[0], prod[1]): [] for prod in itertools.product(partisanships, ideologies) }
  believed_by_cit_p_i = deepcopy(heard_by_cit_p_i)
  # This will be a dict of { (citizen_p, citizen_i): { (message_p, message_i):
  # [ list of (p,i) message heard ] } } - i.e. a dictionary of message types
  # heard by political group
  heard_by_pi_given_cit_pi = { (prod[0], prod[1]): deepcopy(heard_by_cit_p_i) for prod in itertools.product(partisanships, ideologies) }
  believed_by_pi_given_cit_pi = { (prod[0], prod[1]): deepcopy(heard_by_cit_p_i) for prod in itertools.product(partisanships, ideologies) }

  virus_bel_counts = virus_belief_keyed_dict.copy()
  virus_bel_totals = { (prod[0], prod[1], prod[2]): [] for prod in itertools.product(virus_believe_vals, repeat=3) }

  ending_beliefs_by_p_i = { (prod[0], prod[1]): virus_belief_keyed_dict.copy() for prod in itertools.product(partisanships, ideologies) }
  # Similarly to above, this will be a dict of { (cit_p, cit_i): {
  # (virus_beliefs...): [ list of per-citizen (vg,vs,vd) messages heard ] } }
  heard_by_virus_bel_given_pi = { (prod[0], prod[1]): deepcopy(virus_bel_totals) for prod in itertools.product(partisanships, ideologies) }
  believed_by_virus_bel_given_pi = { (prod[0], prod[1]): deepcopy(virus_bel_totals) for prod in itertools.product(partisanships, ideologies) }

  for cit in cit_df.iterrows():
    citizen = cit[1]
    cit_id = int(citizen['who'].replace('"',''))
    brain = nlogo_mixed_list_to_dict(citizen['brain'])
    stats = per_cit_stats[cit_id]

    pi_tup = pi_tuple(brain)
    virus_tup = virus_tuple(brain)

    total_by_i[int(brain['I'])] += 1
    total_by_p[int(brain['P'])] += 1
    total_by_p_i[pi_tup] += 1
    heard_by_cit_p_i[pi_tup].append(stats['total_heard'])
    believed_by_cit_p_i[pi_tup].append(stats['total_believed'])
    for message_pi_tup in stats['heard_by_p_i'].keys():
      heard_by_pi_given_cit_pi[pi_tup][message_pi_tup].append(stats['heard_by_p_i'][message_pi_tup])
      believed_by_pi_given_cit_pi[pi_tup][message_pi_tup].append(stats['believed_by_p_i'][message_pi_tup])

    virus_bel_counts[virus_tup] += 1
    ending_beliefs_by_p_i[pi_tup][virus_tup] += 1
    for message_virus_tup in stats['heard_by_virus_bel'].keys():
      heard_by_virus_bel_given_pi[pi_tup][message_virus_tup].append(stats['heard_by_virus_bel'][message_virus_tup])
      believed_by_virus_bel_given_pi[pi_tup][message_virus_tup].append(stats['believed_by_virus_bel'][message_virus_tup])
  
  heard_sum_by_p_i = { pi: summary_statistics(heard_by_cit_p_i[pi]) for pi in heard_by_cit_p_i.keys() }
  believed_sum_by_p_i = { pi: summary_statistics(believed_by_cit_p_i[pi]) for pi in believed_by_cit_p_i.keys() }

  heard_sum_by_pi_given_pi = pi_keyed_dict.copy()
  for pi in heard_by_pi_given_cit_pi.keys():
    entry = heard_by_pi_given_cit_pi[pi]
    heard_sum_by_pi_given_pi[pi] = { cit_pi: summary_statistics(entry[cit_pi]) for cit_pi in entry.keys() }

  believed_sum_by_pi_given_pi = pi_keyed_dict.copy()
  for pi in believed_by_pi_given_cit_pi.keys():
    entry = believed_by_pi_given_cit_pi[pi]
    believed_sum_by_pi_given_pi[pi] = { cit_pi: summary_statistics(entry[cit_pi]) for cit_pi in entry.keys() }

  heard_sum_by_virus_given_pi = pi_keyed_dict.copy()
  for pi in heard_by_virus_bel_given_pi.keys():
    entry = heard_by_virus_bel_given_pi[pi]
    heard_sum_by_virus_given_pi[pi] = { virus_bel: summary_statistics(entry[virus_bel]) for virus_bel in entry.keys() }

  believed_sum_by_virus_given_pi = pi_keyed_dict.copy()
  for pi in believed_by_virus_bel_given_pi.keys():
    entry = believed_by_virus_bel_given_pi[pi]
    believed_sum_by_virus_given_pi[pi] = { virus_bel: summary_statistics(entry[virus_bel]) for virus_bel in entry.keys() }
  
  stats_given_pi = pi_keyed_dict.copy()
  for pi in stats_given_pi.keys():
    stats_given_pi[pi] = {}
    stats_given_pi[pi]['n'] = total_by_p_i[pi]
    stats_given_pi[pi]['total_heard'] = heard_sum_by_p_i[pi]
    stats_given_pi[pi]['total_believed'] = believed_sum_by_p_i[pi]
    stats_given_pi[pi]['ending_beliefs'] = ending_beliefs_by_p_i[pi]
    stats_given_pi[pi]['heard_stats_by_pi'] = heard_sum_by_pi_given_pi[pi]
    stats_given_pi[pi]['believed_stats_by_pi'] = believed_sum_by_pi_given_pi[pi]
    stats_given_pi[pi]['heard_stats_by_virus'] = heard_sum_by_virus_given_pi[pi]
    stats_given_pi[pi]['believed_stats_by_virus'] = believed_sum_by_virus_given_pi[pi]

  return stats_given_pi

'''
Generate some statistics for each citizen in the simulation report data.
Measures that are reported:
- Total messages heard and believed
- Believed/heard ratio
- Messages heard & believed by (partisan,ideology) pair
- Messages heard & believed by virus-belief combination
'''
def per_citizen_stats(cit_id, messages_df, heard_df, believed_df):
  cit_stats = {}
  cit_stats['total_heard'] = len(heard_df)
  cit_stats['total_believed'] = len(believed_df)
  cit_stats['bel_heard_ratio'] = cit_stats['total_believed']/cit_stats['total_heard']

  partisanships = list(attrs_as_array(Attributes.P))
  ideologies = list(attrs_as_array(Attributes.I))
  heard_by_p_i = { (prod[0], prod[1]): 0 for prod in itertools.product(partisanships, ideologies) }

  for row in heard_df.iterrows():
    heard = row[1]
    m_id = int(heard['message_id'])
    message = messages_df[messages_df['id'] == m_id]
    heard_by_p_i[pi_tuple(message)] += 1

  # (P, I) tuples
  believed_by_p_i = { (prod[0], prod[1]): 0 for prod in itertools.product(partisanships, ideologies) }
  for row in believed_df.iterrows():
    believed = row[1]
    m_id = int(believed['message_id'])
    message = messages_df[messages_df['id'] == m_id]
    believed_by_p_i[pi_tuple(message)] += 1
  cit_stats['heard_by_p_i'] = heard_by_p_i
  cit_stats['believed_by_p_i'] = believed_by_p_i

  # (VG, VG, VD) tuples
  virus_believe_vals = [ -1, 0, 1 ]
  heard_by_virus_bel = { (prod[0], prod[1], prod[2]): 0 for prod in itertools.product(virus_believe_vals, repeat=3) }
  for row in heard_df.iterrows():
    heard = row[1]
    m_id = int(heard['message_id'])
    message = messages_df[messages_df['id'] == m_id]
    heard_by_virus_bel[virus_tuple(message)] += 1
  believed_by_virus_bel = { (prod[0], prod[1], prod[2]): 0 for prod in itertools.product(virus_believe_vals, repeat=3) }
  for row in believed_df.iterrows():
    believed = row[1]
    m_id = int(believed['message_id'])
    message = messages_df[messages_df['id'] == m_id]
    believed_by_virus_bel[virus_tuple(message)] += 1
  cit_stats['heard_by_virus_bel'] = heard_by_virus_bel
  cit_stats['believed_by_virus_bel'] = believed_by_virus_bel

  return cit_stats

def group_stats_by_attr(group_stats, attr):
  return { pi: val[attr] for pi,val in group_stats.items() }

'''
Return a tuple of partisanship and ideology attributes from a given object.

:param obj: Some object to fetch parameters from.
'''
def pi_tuple(obj): return (int(obj['P']),int(obj['I']))

'''
Return a tuple of virus-related beliefs from a given object.

:param obj: Some object to fetch parameters from.
'''
def virus_tuple(obj): return (int(obj['VG']),int(obj['VS']),int(obj['VD']))

def plot_stats_means(stats_data, title, path):
  plot_and_save_series({ key: val[0] for key,val in stats_data.items() }, title, path, 'bar')

def pi_data_charts(stats_data, attr, replace, title_w_replace, path_w_replace):
  partisanships = list(attrs_as_array(Attributes.P))
  ideologies = list(attrs_as_array(Attributes.I))
  pi_keys = { (prod[0], prod[1]): 0 for prod in itertools.product(partisanships, ideologies) }
  for key in pi_keys:
    plot_stats_means(stats_data[key][attr], title_w_replace.replace(replace, str(key)), path_w_replace.replace(replace, f'{key[0]}-{key[1]}'))

def corr_multi_data(multi_data_1, multi_data_2, method='pearson'):
  '''
  Calculate correlations between two sets of multi data.

  :param multi_data_1: A first set of data over multiple simulation runs, keyed by agent belief value.
  :param multi_data_2: A second set of data over multiple simulation runs, keyed by agent belief value.

  :return: Correlation values per belief value.
  '''
  m1_means = { key: multi_data_1[key].mean(0) for key in multi_data_1 }
  m2_means = { key: multi_data_2[key].mean(0) for key in multi_data_2 }

  rs = {}
  for key in multi_data_1:
    df = pd.DataFrame({ 'data1': m1_means[key], 'data2': m2_means[key] })
    # Uncomment if you need to investigate the df
    # rs[key] = {}
    # rs[key]['df'] = df
    # rs[key]['corr'] = df.corr(method=method).iloc[0,1]
    rs[key] = df.corr(method=method).iloc[0,1]
  return rs

def aggregate_corr(corr_by_bel):
  '''
  Generate an average correlation across correlations by belief value.

  :param corr_by_bel: A dictionary keyed by belief value of correlation values.
  '''
  non_nan = np.array(list(corr_by_bel.values()))
  non_nan = non_nan[np.logical_not(np.isnan(non_nan))]
  return non_nan.sum() / len(non_nan)

def chi_sq_test_multi_data(multi_data_1, multi_data_2, N):
  '''
  Perform a chi squared test on two sets of multi data for each timestep in the simulation data. 

  NOTE: This converts agent population percentages to total numbers and pads by 1
  in order to circumnavigate sampling 0 agents.

  :param multi_data_1: A first set of data over multiple simulation runs, keyed by agent belief value.
  :param multi_data_2: A second set of data over multiple simulation runs, keyed by agent belief value.
  :param N: The number of agents in the simulation.

  :returns: Returns the chi2 timeseries data.
  '''

  m1_means = [ multi_data_1[key].mean(0) for key in multi_data_1 ]
  m2_means = [ multi_data_2[key].mean(0) for key in multi_data_2 ]

  data = []
  for timestep in range(len(m1_means[0])):
    data.append([])
    # Append on lists of the values for each belief at timestep t 
    data[timestep].append([ m1_means[bel][timestep] for bel in range(len(m1_means)) ])
    data[timestep].append([ m2_means[bel][timestep] for bel in range(len(m2_means)) ])
  
  for data_t in data:
    for i in range(len(data_t[0])):
      data_t[0][i] = round(N * data_t[0][i] + 1)
      data_t[1][i] = round(N * data_t[1][i] + 1)
  
  chi2_data = [ chi2_contingency(data_t) for data_t in data ]
  # TODO: CHANGE THIS BACK
  # return chi2_data
  return (data, chi2_data)

def chi_sq_global(chi2_data):
  '''
  Convert a timeseries of chi squared test data into a global measure of how many
  entries in the time series are statistically independent. Higher values indicate
  higher levels of independence.

  :param chi2_data: An array of timeseries chi squared data from the scipy test.
  '''
  data = np.array([ el[1] for el in chi2_data ])
  return (data <= 0.05).sum() / len(data)


def plot_chi_sq_data(chi2_data, props, title, out_path, out_filename):
  '''
  Plot a time series calculation of chi squared measures per timestep.

  :param chi2_data: The timeseries data from running chi squared stats on belief data.
  :param props: The simulation properties to pull time data from.
  :param title: Text to title the plot with.
  '''
  # series = pd.Series(data)
  fig, (ax) = plt.subplots(1, figsize=(8,6))
  # ax, ax2 = fig.add_subplot(2)
  ax.set_ylim([0, 1.0])
  y_min = 0
  y_max = 1.0
  x_min = int(props['x min'])
  x_max = int(props['x max'])
  plt.yticks(np.arange(y_min, y_max+0.2, step=0.05))
  plt.xticks(np.arange(x_min, x_max+10, step=10))
  ax.set_ylabel("p value")
  ax.set_xlabel("Time Step")
  ax.set_title(f'{title}')
 
  ax.plot([ data[1] for data in chi2_data ])
  plt.savefig(f'{out_path}/{out_filename}')
  plt.close()

"""
##################
EXPERIMENT-SPECIFIC
ANALYSIS
##################
"""

class PLOT_TYPES(Enum):
  LINE = 0
  STACK = 1
  HISTOGRAM = 2

def process_exp_outputs(param_combos, plots, path):
  '''
  Process the output of a NetLogo experiment, aggregating all results
  over simulation runs and generating plots for them according to
  all the parameter combinations denoted in param_combos.
  
  :param param_combos: A list of parameters where their values are
  lists (e.g. [ ['simple','complex'], ['default', 'gradual'] ])
  :param plots: A list of dictionaries keyed by the name of the NetLogo
  plot to process, with value of a list of PLOT_TYPE
  (e.g. { 'polarization': [PLOT_TYPES.LINE], 'agent-beliefs': [...] })
  :param path: The root path to begin processing in.
  '''
  combos = []
  for combo in itertools.product(*param_combos):
    combos.append(combo)

  if not os.path.isdir(f'{path}/results'):
    os.mkdir(f'{path}/results')

  for combo in combos:
    for (plot_name, plot_types) in plots.items():
      # print(plot_name, plot_types)
      (multi_data, props, model_params) = process_multi_chart_data(f'{path}/{"/".join(combo)}', plot_name)
      # If there was no error processing the data
      if multi_data != -1:
        plot_multi_chart_data(plot_types, multi_data, props, f'{path}/results', f'{"-".join(combo)}_{plot_name}-agg-chart')

def process_select_exp_outputs(param_combos, plots, path, results_dir):
  '''
  Process some of the output of a NetLogo experiment, aggregating specified 
  results over simulation runs and generating plots for them according to
  all the parameter combinations denoted in param_combos.
  
  :param param_combos: A list of selected parameter values as lists
  :param plots: A list of dictionaries keyed by the name of the NetLogo
  plot to process, with value of a list of PLOT_TYPE
  (e.g. { 'polarization': [PLOT_TYPES.LINE], 'agent-beliefs': [...] })
  :param path: The root path to begin processing in.
  '''
  if not os.path.isdir(f'{path}/{results_dir}'):
    os.mkdir(f'{path}/{results_dir}')

  for combo in param_combos:
    for (plot_name, plot_types) in plots.items():
      # print(plot_name, plot_types)
      (multi_data, props, model_params) = process_multi_chart_data(f'{path}/{"/".join(combo)}', plot_name)
      # If there was no error processing the data
      if multi_data != -1:
        plot_multi_chart_data(plot_types, multi_data, props, f'{path}/{results_dir}', f'{"-".join(combo)}_{plot_name}-agg-chart')

def get_all_message_multidata(param_combos, path):
  combos = []
  for combo in itertools.product(*param_combos):
    combos.append(combo)

  multi_datas = {}
  for combo in combos:
    multi_message_data = process_multi_message_data(f'{path}/{"/".join(combo)}')
    multi_datas[combo] = multi_message_data
  return multi_datas

def process_all_message_multidata(param_combos, path):
  combos = []
  for combo in itertools.product(*param_combos):
    combos.append(combo)

  multi_datas = {}
  for combo in combos:
    multi_message_data = process_multi_message_data(f'{path}/{"/".join(combo)}')
    plot_multi_message_data((combo, multi_message_data), f'{path}/results', False)
    multi_datas[combo] = multi_message_data
  return multi_datas

def get_all_multidata(param_combos, params, plots, path):
  combos = []
  for combo in itertools.product(*param_combos):
    combos.append(combo)

  multi_datas = {}
  for combo in combos:
    for (plot_name, plot_types) in plots.items():
      # print(plot_name, plot_types)
      (multi_data, props, model_params) = process_multi_chart_data(f'{path}/{"/".join(combo)}', plot_name)
      multi_datas[(combo,plot_name)] = multi_data
      multi_datas['params'] = params
  return multi_datas

def missing_values_in_multidata(multidata):
  return { key: val for key,val in multidata.items() if val == -1 }

def multidata_as_dataframe(multidata, columns):
  df = pd.DataFrame(columns=(columns+['measure','pen_name','run','data']))
  for (param_measure,data) in multidata.items():
    if param_measure == 'params':
      continue
    # Something failed to read/didn't exist
    if data == -1:
      continue
    param_combo = param_measure[0]
    measure = param_measure[1]
    for (pen_name,runs_data) in data.items():
      for run in range(len(runs_data)):
        df.loc[len(df.index)] = list(param_combo) + [measure,pen_name,run,runs_data[run]]
  return df

def mean_multidata_as_dataframe(multidata, columns):
  df = pd.DataFrame(columns=(columns+['measure','pen_name','data']))
  for (param_measure,data) in multidata.items():
    if param_measure == 'params':
      continue
    # Something failed to read/didn't exist
    if data == -1:
      continue
    param_combo = param_measure[0]
    measure = param_measure[1]
    for (pen_name,runs_data) in data.items():
      df.loc[len(df.index)] = list(param_combo) + [measure,pen_name,runs_data]
  return df

def dataframe_as_multidata(df):
  multidata = {}
  added_columns = ['measure','pen_name','run','data']
  for row in df.itertuples():
    param_columns = tuple([ col for col in df.columns if col not in added_columns ])
    param_col_values = tuple([ getattr(row,col) for col in param_columns ])
    measure = row.measure
    pen_name = row.pen_name
    key = (param_col_values,measure)
    if key not in multidata:
      multidata[key] = { }
    
    if pen_name not in multidata[key]:
      multidata[key][pen_name] = np.array([row.data])
    else:
      data = multidata[key][pen_name]
      multidata[key][pen_name] = np.vstack([data,row.data])
          # means[key] = np.vstack([means[key], data_vector])
  return multidata

def read_polarization_dataframe(path):
  df = pd.read_csv(path)
  for i in range(len(df)):
    raw_data = df.iloc[i]['data']
    df.at[i,'data'] = np.fromstring(raw_data[1:-1].replace('\n','').replace('0. ','0 '),sep=' ')
  return df

def process_top_exp_results(top_df, param_order, path, results_dir):
  top_param_combos = [
    [ str(top_df[col].iloc[i]) for col in param_order ]
    for i in range(len(top_df))
  ]
  print(top_df)
  print(top_param_combos)
  process_select_exp_outputs(
    top_param_combos,
    {'percent-agent-beliefs': [PLOT_TYPES.LINE, PLOT_TYPES.STACK],
    'opinion-timeseries': [PLOT_TYPES.LINE]},
    path,
    results_dir)

def process_simple_contagion_param_sweep_ER_top(top_df, path, results_dir):
  param_order = ['er_p','simple_spread_chance','repetition']
  process_top_exp_results(top_df, param_order, path, results_dir)

def process_cognitive_contagion_param_sweep_ER_top(top_df, path, results_dir):
  param_order = ['er_p','cognitive_translate','cognitive_exponent','repetition']
  process_top_exp_results(top_df, param_order, path, results_dir)

def process_simple_contagion_param_sweep_ER_test(path):
  simple_spread_chance = ['0.01','0.05','0.1','0.25','0.5','0.75']
  er_p = ['0.05','0.1','0.25','0.5']
  repetition = list(map(str, range(4)))
  process_exp_outputs(
    [er_p,simple_spread_chance,repetition],
    {'percent-agent-beliefs': [PLOT_TYPES.LINE, PLOT_TYPES.STACK],
    'opinion-timeseries': [PLOT_TYPES.LINE]},
    path)

def get_simple_contagion_param_sweep_ER_test_multidata(path):
  simple_spread_chance = ['0.01','0.05','0.1','0.25','0.5','0.75']
  er_p = ['0.05','0.1','0.25','0.5']
  repetition = list(map(str, range(4)))
  measure_multidata = get_all_multidata(
    [er_p,simple_spread_chance,repetition],
    ['er_p','simple_spread_chance','repetition'],
    {'percent-agent-beliefs': [PLOT_TYPES.LINE, PLOT_TYPES.STACK],
    'opinion-timeseries': [PLOT_TYPES.LINE]},
    path)
  return measure_multidata

def get_simple_contagion_param_sweep_ER_multidata(path):
  simple_spread_chance = ['0.01','0.05','0.1','0.25','0.5','0.75']
  er_p = ['0.05','0.1','0.25','0.5']
  repetition = list(map(str, range(10)))
  measure_multidata = get_all_multidata(
    [er_p,simple_spread_chance,repetition],
    ['er_p','simple_spread_chance','repetition'],
    {'percent-agent-beliefs': [PLOT_TYPES.LINE, PLOT_TYPES.STACK],
    'opinion-timeseries': [PLOT_TYPES.LINE]},
    path)
  return measure_multidata

def get_cognitive_contagion_param_sweep_ER_multidata(path):
  cognitive_exponent = ['1','2','3','4','5']
  cognitive_translate = ['0','1','2','3']
  er_p = ['0.05','0.1','0.25','0.5']
  repetition = list(map(str, range(10)))
  measure_multidata = get_all_multidata(
    [er_p,cognitive_translate,cognitive_exponent,repetition],
    ['er_p','cognitive_translate','cognitive_exponent','repetition'],
    {'percent-agent-beliefs': [PLOT_TYPES.LINE, PLOT_TYPES.STACK],
    'opinion-timeseries': [PLOT_TYPES.LINE]},
    path)
  return measure_multidata

def process_simple_contagion_param_sweep_WS_top(top_df, path, results_dir):
  param_order = ['ws_p','ws_k','simple_spread_chance','repetition']
  process_top_exp_results(top_df, param_order, path, results_dir)

def process_cognitive_contagion_param_sweep_WS_top(top_df, path, results_dir):
  param_order = ['ws_p','ws_k','cognitive_translate','cognitive_exponent','repetition']
  process_top_exp_results(top_df, param_order, path, results_dir)

def get_simple_contagion_param_sweep_WS_multidata(path):
  simple_spread_chance = ['0.01','0.05','0.1','0.25','0.5','0.75']
  ws_p = ['0.1','0.25','0.5']
  ws_k = ['2','3','5','10','15']
  repetition = list(map(str, range(10)))
  measure_multidata = get_all_multidata(
    [ws_p,ws_k,simple_spread_chance,repetition],
    ['ws_p','ws_k','simple_spread_chance','repetition'],
    {'percent-agent-beliefs': [PLOT_TYPES.LINE, PLOT_TYPES.STACK],
    'opinion-timeseries': [PLOT_TYPES.LINE]},
    path)
  return measure_multidata

def get_cognitive_contagion_param_sweep_WS_multidata(path):
  cognitive_exponent = ['1','2','3','4','5']
  cognitive_translate = ['0','1','2','3']
  ws_p = ['0.1','0.25','0.5']
  ws_k = ['2','3','5','10','15']
  repetition = list(map(str, range(10)))
  measure_multidata = get_all_multidata(
    [ws_p,ws_k,cognitive_translate,cognitive_exponent,repetition],
    ['ws_p','ws_k','cognitive_translate','cognitive_exponent','repetition'],
    {'percent-agent-beliefs': [PLOT_TYPES.LINE, PLOT_TYPES.STACK],
    'opinion-timeseries': [PLOT_TYPES.LINE]},
    path)
  return measure_multidata

def process_simple_contagion_param_sweep_BA_top(top_df, path, results_dir):
  param_order = ['ba_m','simple_spread_chance','repetition']
  process_top_exp_results(top_df, param_order, path, results_dir)

def process_cognitive_contagion_param_sweep_BA_top(top_df, path, results_dir):
  param_order = ['ba_m','cognitive_translate','cognitive_exponent','repetition']
  process_top_exp_results(top_df, param_order, path, results_dir)

def get_simple_contagion_param_sweep_BA_multidata(path):
  simple_spread_chance = ['0.01','0.05','0.1','0.25','0.5','0.75']
  ba_m = ['3','5','10','15']
  repetition = list(map(str, range(10)))
  measure_multidata = get_all_multidata(
    [ba_m,simple_spread_chance,repetition],
    ['ba-m','simple_spread_chance','repetition'],
    {'percent-agent-beliefs': [PLOT_TYPES.LINE, PLOT_TYPES.STACK],
    'opinion-timeseries': [PLOT_TYPES.LINE]},
    path)
  return measure_multidata

def get_cognitive_contagion_param_sweep_BA_multidata(path):
  cognitive_exponent = ['1','2','3','4','5']
  cognitive_translate = ['0','1','2','3']
  ba_m = ['3','5','10','15']
  repetition = list(map(str, range(10)))
  measure_multidata = get_all_multidata(
    [ba_m,cognitive_translate,cognitive_exponent,repetition],
    ['ba-m','cognitive_translate','cognitive_exponent','repetition'],
    {'percent-agent-beliefs': [PLOT_TYPES.LINE, PLOT_TYPES.STACK],
    'opinion-timeseries': [PLOT_TYPES.LINE]},
    path)
  return measure_multidata

def read_gallup_data_into_dict(path):
  gallup_data = pd.read_csv(path)
  gallup_data.drop(columns=['Unnamed: 0'], inplace=True)
  gallup_dict = { col: np.array(gallup_data[col]) for col in gallup_data.columns }
  return gallup_dict

def metrics_for_simple_contagion_param_sweep_ER_test(path):
  gallup_dict = read_gallup_data_into_dict('../labeled-data/public/gallup-polling.csv')
  columns = ['er_p','simple_spread_chance','repetition']
  measure = 'opinion-timeseries'
  multidata = get_simple_contagion_param_sweep_ER_test_multidata(path)
  all_run_metrics = timeseries_similarity_for_all_runs(multidata, measure, columns, gallup_dict)
  mean_metrics = timeseries_similarity_for_mean_runs(multidata, measure, columns, gallup_dict)
  return all_run_metrics, mean_metrics

def metrics_for_simple_contagion_param_sweep_ER(path):
  gallup_dict = read_gallup_data_into_dict('../labeled-data/public/gallup-polling.csv')
  columns = ['er_p','simple_spread_chance','repetition']
  measure = 'opinion-timeseries'
  multidata = get_simple_contagion_param_sweep_ER_multidata(path)
  all_run_metrics = timeseries_similarity_for_all_runs(multidata, measure, columns, gallup_dict)
  mean_metrics = timeseries_similarity_for_mean_runs(multidata, measure, columns, gallup_dict)
  return all_run_metrics, mean_metrics

def metrics_for_simple_contagion_param_sweep_WS(path):
  gallup_dict = read_gallup_data_into_dict('../labeled-data/public/gallup-polling.csv')
  columns = ['ws_p','ws_k','simple_spread_chance','repetition']
  measure = 'opinion-timeseries'
  multidata = get_simple_contagion_param_sweep_WS_multidata(path)
  all_run_metrics = timeseries_similarity_for_all_runs(multidata, measure, columns, gallup_dict)
  mean_metrics = timeseries_similarity_for_mean_runs(multidata, measure, columns, gallup_dict)
  return all_run_metrics, mean_metrics

def metrics_for_simple_contagion_param_sweep_BA(path):
  gallup_dict = read_gallup_data_into_dict('../labeled-data/public/gallup-polling.csv')
  columns = ['ba_m','simple_spread_chance','repetition']
  measure = 'opinion-timeseries'
  multidata = get_simple_contagion_param_sweep_BA_multidata(path)
  all_run_metrics = timeseries_similarity_for_all_runs(multidata, measure, columns, gallup_dict)
  mean_metrics = timeseries_similarity_for_mean_runs(multidata, measure, columns, gallup_dict)
  return all_run_metrics, mean_metrics

def metrics_for_cognitive_contagion_param_sweep_ER(path):
  gallup_dict = read_gallup_data_into_dict('../labeled-data/public/gallup-polling.csv')
  columns = ['er_p','cognitive_translate','cognitive_exponent','repetition']
  measure = 'opinion-timeseries'
  multidata = get_cognitive_contagion_param_sweep_ER_multidata(path)
  all_run_metrics = timeseries_similarity_for_all_runs(multidata, measure, columns, gallup_dict)
  mean_metrics = timeseries_similarity_for_mean_runs(multidata, measure, columns, gallup_dict)
  return all_run_metrics, mean_metrics

def metrics_for_cognitive_contagion_param_sweep_WS(path):
  gallup_dict = read_gallup_data_into_dict('../labeled-data/public/gallup-polling.csv')
  columns = ['ws_p','ws_k','cognitive_translate','cognitive_exponent','repetition']
  measure = 'opinion-timeseries'
  multidata = get_cognitive_contagion_param_sweep_WS_multidata(path)
  all_run_metrics = timeseries_similarity_for_all_runs(multidata, measure, columns, gallup_dict)
  mean_metrics = timeseries_similarity_for_mean_runs(multidata, measure, columns, gallup_dict)
  return all_run_metrics, mean_metrics

def metrics_for_cognitive_contagion_param_sweep_BA(path):
  gallup_dict = read_gallup_data_into_dict('../labeled-data/public/gallup-polling.csv')
  columns = ['ba_m','cognitive_translate','cognitive_exponent','repetition']
  measure = 'opinion-timeseries'
  multidata = get_cognitive_contagion_param_sweep_BA_multidata(path)
  all_run_metrics = timeseries_similarity_for_all_runs(multidata, measure, columns, gallup_dict)
  mean_metrics = timeseries_similarity_for_mean_runs(multidata, measure, columns, gallup_dict)
  return all_run_metrics, mean_metrics

def top_matches_for_metrics(metrics_df):
  ranked = metrics_df.sort_values(by=['mape','pearson'], ascending=[True,False])
  # ranked = metrics_df.sort_values(by=['pearson','mape'], ascending=[False,True])
  return ranked.head(10)

def mean_multidata(multidata):
  multi_data_has_multiple = lambda multi_data_entry: type(multi_data_entry[0]) == type(np.array(0)) and len(multi_data_entry) > 1
  mean_multidata = {
    # param_measure[0] is the parameter combo tuple
    param_measure: {
      pen_name: (multidata[param_measure][pen_name].mean(0) if multi_data_has_multiple(multidata[param_measure][pen_name]) else multidata[param_measure][pen_name]) for pen_name in multidata[param_measure].keys() if multidata[param_measure] != -1
    } for param_measure in multidata.keys() if param_measure != 'params'
  }
  return mean_multidata

def multidata_means_as_df(multidata, columns):
  mean_data = mean_multidata(multidata)
  df_mean = mean_multidata_as_dataframe(mean_data, columns)
  return df_mean

def timeseries_similarity_for_all_runs(multidata, measure, columns, target_data):
  multidata_measure = { key: val for key, val in multidata.items() if key[1] == measure }
  df = multidata_as_dataframe(multidata_measure, columns)
  columns = columns.copy() + ['run']
  return timeseries_similarity_scores_for_simulations(df, columns, target_data)

def timeseries_similarity_for_mean_runs(multidata, measure, columns, target_data):
  multidata_measure = { key: val for key, val in multidata.items() if key[1] == measure }
  df = multidata_means_as_df(multidata_measure, columns)
  return timeseries_similarity_scores_for_simulations(df, columns, target_data)

def timeseries_similarity_scores_for_simulations(df, columns, target_data):
  column_values = { col: df[col].unique() for col in columns }
  param_combos = []
  for combo in itertools.product(*list(column_values.values())):
    param_combos.append(combo)

  metrics = { 
    'pearson': lambda simulated, empirical: np.corrcoef(simulated, empirical)[0,1],
    'euclidean': lambda simulated, empirical: np.sqrt(np.sum((empirical - simulated) ** 2)),
    'mape': lambda simulated, empirical: np.mean(np.abs((empirical - simulated) / empirical))
  }
  df_comparison_results = pd.DataFrame(columns=columns + list(metrics.keys()))

  for param_vals in param_combos:
    query = ''
    for i in range(len(columns)):
      param = columns[i]
      val = param_vals[i]
      str_val = f'"{val}"'
      query += f"{param}=={val if type(val) != str else str_val} and "
    query = query[:-5]
    df_rows = df.query(query)
    if len(df_rows) > 0:
      timeseries_by_pen_name = { row[1]['pen_name']: row[1]['data'] for row in df_rows.iterrows() }
      # This is an assumption made -- to take the mean of the scores of each
      # of the separate lines and have that be the aggregate score
      # NOTE: Slicing the dataframe to only 64 entries is to account for an
      # error in earlier simulations where they were run for 74 time steps;
      # slicing to a smaller value does NOT change the results
      metric_results = [ np.array([ metric_fn(timeseries_by_pen_name[key][:64], target_data[key]) for key in timeseries_by_pen_name.keys() ]).mean() for metric_fn in metrics.values() ]
      df_comparison_results.loc[len(df_comparison_results)] = [ df_rows.iloc[0][param] for param in columns ] + metric_results
    else:
      print(f'Unable to take metric for nonexistant rows for param combo: {param_vals}')

  return df_comparison_results

def process_all_cognitive_exp_metrics():
  er_metrics = metrics_for_cognitive_contagion_param_sweep_ER(f'{DATA_DIR}/cognitive-contagion-sweep-ER')
  ws_metrics = metrics_for_cognitive_contagion_param_sweep_WS(f'{DATA_DIR}/cognitive-contagion-sweep-WS')
  ba_metrics = metrics_for_cognitive_contagion_param_sweep_BA(f'{DATA_DIR}/cognitive-contagion-sweep-BA')

  er_metrics[0].to_csv('./data/analyses/cognitive-er-all.csv')
  er_metrics[1].to_csv('./data/analyses/cognitive-er-mean.csv')
  ws_metrics[0].to_csv('./data/analyses/cognitive-ws-all.csv')
  ws_metrics[1].to_csv('./data/analyses/cognitive-ws-mean.csv')
  ba_metrics[0].to_csv('./data/analyses/cognitive-ba-all.csv')
  ba_metrics[1].to_csv('./data/analyses/cognitive-ba-mean.csv')

  er_all_top = top_matches_for_metrics(er_metrics[0])
  er_mean_top = top_matches_for_metrics(er_metrics[1])
  ws_all_top = top_matches_for_metrics(ws_metrics[0])
  ws_mean_top = top_matches_for_metrics(ws_metrics[1])
  ba_all_top = top_matches_for_metrics(ba_metrics[0])
  ba_mean_top = top_matches_for_metrics(ba_metrics[1])

  er_all_top.to_csv('./data/analyses/cognitive-er-all_top.csv')
  er_mean_top.to_csv('./data/analyses/cognitive-er-mean_top.csv')
  ws_all_top.to_csv('./data/analyses/cognitive-ws-all_top.csv')
  ws_mean_top.to_csv('./data/analyses/cognitive-ws-mean_top.csv')
  ba_all_top.to_csv('./data/analyses/cognitive-ba-all_top.csv')
  ba_mean_top.to_csv('./data/analyses/cognitive-ba-mean_top.csv')

  process_cognitive_contagion_param_sweep_ER_top(er_all_top, f'{DATA_DIR}/cognitive-contagion-sweep-ER', 'results-all')
  process_cognitive_contagion_param_sweep_WS_top(ws_all_top, f'{DATA_DIR}/cognitive-contagion-sweep-WS', 'results-all')
  process_cognitive_contagion_param_sweep_BA_top(ba_all_top, f'{DATA_DIR}/cognitive-contagion-sweep-BA', 'results-all')
  process_cognitive_contagion_param_sweep_ER_top(er_mean_top, f'{DATA_DIR}/cognitive-contagion-sweep-ER', 'results-mean')
  process_cognitive_contagion_param_sweep_WS_top(ws_mean_top, f'{DATA_DIR}/cognitive-contagion-sweep-WS', 'results-mean')
  process_cognitive_contagion_param_sweep_BA_top(ba_mean_top, f'{DATA_DIR}/cognitive-contagion-sweep-BA', 'results-mean')

def process_all_simple_exp_metrics():
  data_path = './data/analyses'
  er_metrics = []
  if exists(f'{data_path}/simple-er-all.csv') and exists(f'{data_path}/simple-er-mean.csv'):
    print('Read in ER metric data')
    er_metrics.append(pd.read_csv(f'{data_path}/simple-er-all.csv'))
    er_metrics.append(pd.read_csv(f'{data_path}/simple-er-mean.csv'))
    er_metrics[0].drop(columns=['Unnamed: 0'], inplace=True)
    er_metrics[1].drop(columns=['Unnamed: 0'], inplace=True)
  else:
    er_metrics = metrics_for_simple_contagion_param_sweep_ER(f'{DATA_DIR}/simple-contagion-sweep-ER')
    er_metrics[0].to_csv(f'{data_path}/simple-er-all.csv')
    er_metrics[1].to_csv(f'{data_path}/simple-er-mean.csv')

  ws_metrics = []
  if exists(f'{data_path}/simple-ws-all.csv') and exists(f'{data_path}/simple-ws-mean.csv'):
    print('Read in WS metric data')
    ws_metrics.append(pd.read_csv(f'{data_path}/simple-ws-all.csv'))
    ws_metrics.append(pd.read_csv(f'{data_path}/simple-ws-mean.csv'))
    ws_metrics[0].drop(columns=['Unnamed: 0'], inplace=True)
    ws_metrics[1].drop(columns=['Unnamed: 0'], inplace=True)
  else:
    ws_metrics = metrics_for_simple_contagion_param_sweep_WS(f'{DATA_DIR}/simple-contagion-sweep-WS')
    ws_metrics[0].to_csv(f'{data_path}/simple-ws-all.csv')
    ws_metrics[1].to_csv(f'{data_path}/simple-ws-mean.csv')

  ba_metrics = []
  if exists(f'{data_path}/simple-ba-all.csv') and exists(f'{data_path}/simple-ba-mean.csv'):
    print('Read in BA metric data')
    ba_metrics.append(pd.read_csv(f'{data_path}/simple-ba-all.csv'))
    ba_metrics.append(pd.read_csv(f'{data_path}/simple-ba-mean.csv'))
    ba_metrics[0].drop(columns=['Unnamed: 0'], inplace=True)
    ba_metrics[1].drop(columns=['Unnamed: 0'], inplace=True)
  else:
    ba_metrics = metrics_for_simple_contagion_param_sweep_BA(f'{DATA_DIR}/simple-contagion-sweep-BA')
    ba_metrics[0].to_csv(f'{data_path}/simple-ba-all.csv')
    ba_metrics[1].to_csv(f'{data_path}/simple-ba-mean.csv')

  er_all_top = None
  er_mean_top = None
  if exists(f'{data_path}/simple-er-all_top.csv'):
    er_all_top = pd.read_csv(f'{data_path}/simple-er-all_top.csv')
    er_all_top.drop(columns=['Unnamed: 0'], inplace=True)
  else:
    er_all_top = top_matches_for_metrics(er_metrics[0])
    er_all_top.to_csv(f'{data_path}/simple-er-all_top.csv')
  if exists(f'{data_path}/simple-er-mean_top.csv'):
    er_mean_top = pd.read_csv(f'{data_path}/simple-er-mean_top.csv')
    er_mean_top.drop(columns=['Unnamed: 0'], inplace=True)
  else:
    er_mean_top = top_matches_for_metrics(er_metrics[1])
    er_mean_top.to_csv(f'{data_path}/simple-er-mean_top.csv')

  ws_all_top = None
  ws_mean_top = None
  if exists(f'{data_path}/simple-ws-all_top.csv'):
    ws_all_top = pd.read_csv(f'{data_path}/simple-ws-all_top.csv')
    ws_all_top.drop(columns=['Unnamed: 0'], inplace=True)
  else:
    ws_all_top = top_matches_for_metrics(ws_metrics[0])
    ws_all_top.to_csv(f'{data_path}/simple-ws-all_top.csv')
  if exists(f'{data_path}/simple-ws-mean_top.csv'):
    ws_mean_top = pd.read_csv(f'{data_path}/simple-ws-mean_top.csv')
    ws_mean_top.drop(columns=['Unnamed: 0'], inplace=True)
  else:
    ws_mean_top = top_matches_for_metrics(ws_metrics[1])
    ws_mean_top.to_csv(f'{data_path}/simple-ws-mean_top.csv')

  ba_all_top = None
  ba_mean_top = None
  if exists(f'{data_path}/simple-ba-all_top.csv'):
    ba_all_top = pd.read_csv(f'{data_path}/simple-ba-all_top.csv')
    ba_all_top.drop(columns=['Unnamed: 0'], inplace=True)
  else:
    ba_all_top = top_matches_for_metrics(ba_metrics[0])
    ba_all_top.to_csv(f'{data_path}/simple-ba-all_top.csv')
  if exists(f'{data_path}/simple-ba-mean_top.csv'):
    ba_mean_top = pd.read_csv(f'{data_path}/simple-ba-mean_top.csv')
    ba_mean_top.drop(columns=['Unnamed: 0'], inplace=True)
  else:
    ba_mean_top = top_matches_for_metrics(ba_metrics[1])
    ba_mean_top.to_csv(f'{data_path}/simple-ba-mean_top.csv')

  process_simple_contagion_param_sweep_ER_top(er_all_top, f'{DATA_DIR}/simple-contagion-sweep-ER', 'results-all')
  process_simple_contagion_param_sweep_WS_top(ws_all_top, f'{DATA_DIR}/simple-contagion-sweep-WS', 'results-all')
  process_simple_contagion_param_sweep_BA_top(ba_all_top, f'{DATA_DIR}/simple-contagion-sweep-BA', 'results-all')
  process_simple_contagion_param_sweep_ER_top(er_mean_top, f'{DATA_DIR}/simple-contagion-sweep-ER', 'results-mean')
  process_simple_contagion_param_sweep_WS_top(ws_mean_top, f'{DATA_DIR}/simple-contagion-sweep-WS', 'results-mean')
  process_simple_contagion_param_sweep_BA_top(ba_mean_top, f'{DATA_DIR}/simple-contagion-sweep-BA', 'results-mean')

def low_res_sweep_total_analysis(data_dir, data_file):
  return dynamic_model_total_analysis(data_dir, data_file, ['translate','tactic','media_dist','media_n','citizen_dist','zeta_citizen','zeta_media','citizen_memory_len','repetition'])

def low_res_low_media_total_analysis(data_dir, data_file):
  return dynamic_model_total_analysis(data_dir, data_file, ['translate','media_dist','tactic','citizen_dist','zeta_citizen','zeta_media','citizen_memory_len','repetition'])

def static_sweep_large_N_total_analysis(data_dir, data_file):
  return static_model_total_analysis(data_dir, data_file, ['translate','tactic','media_dist','citizen_dist','epsilon','graph_type','ba_m','repetition'])

'''
================
ANALYSES OF RESULTS
================
'''

def static_model_total_analysis(data_dir, data_file, params):
  print('loading polarization data...')
  df = read_polarization_dataframe(f'{data_dir}/{data_file}')
  df = df.drop(columns=['Unnamed: 0'])
  df['pen_name'] = df['pen_name'].astype(str)
  multidata = dataframe_as_multidata(df)
  multidata['params'] = params
  print('loaded polarization data')

  print('starting polarization analysis...')
  polarization_slope = 0.01
  polarization_intercept = 8.5

  polarization_categories = ['polarized','depolarized','remained_polarized','remained_nonpolarized']

  polarization_data = polarization_all_analysis(multidata, polarization_slope, polarization_intercept)
  with open(f'{data_dir}/polarization-results.json','w') as f:
    polarization_results = { category: len(polarization_data[category]) for category in polarization_categories }
    json.dump(polarization_results, f)
  print('finished polarization analysis')
 
  polarization_all_df = polarization_data['polarization_df']

  print('starting polarization stability analysis...')
  polarization_stability_data = polarization_stability_analysis(multidata, polarization_slope, polarization_intercept)
  with open(f'{data_dir}/polarization-stability-diff-parts.json','w') as f:
    json.dump(polarization_stability_data['diff_parts'], f)
  print('finished polarization stability analysis')

  stability_df = polarization_stability_data['stability']
  print('starting polarization results by parameter breakdown...')
  (fe_relative, fe_absolute) = static_polarization_results_by_fragmentation_exposure(polarization_all_df)
  with open(f'{data_dir}/polarization-by-frag-zeta-relative.json','w') as f:
    json.dump(fe_relative, f)
  with open(f'{data_dir}/polarization-by-frag-zeta-absolute.json','w') as f:
    json.dump(fe_absolute, f)
  write_static_polarization_by_fragmentation_exposure(fe_absolute, data_dir, 'polarization-by-frag-zeta-absolute-LATEX.tex')
  write_static_polarization_by_fragmentation_exposure(fe_relative, data_dir, 'polarization-by-frag-zeta-relative-LATEX.tex')

  print('finished breakdown by fragmentation and exposure')

  (td_relative, td_absolute) = polarization_results_by_tactic_distributions(polarization_all_df)
  with open(f'{data_dir}/polarization-all-by-tactic-dist-relative.json','w') as f:
    json.dump(td_relative, f)
  with open(f'{data_dir}/polarization-all-by-tactic-dist-absolute.json','w') as f:
    json.dump(td_absolute, f)
  write_polarization_by_tactic_distribution(td_absolute, data_dir, 'polarization-all-by-tactic-dist-absolute-LATEX.tex')
  write_polarization_by_tactic_distribution(td_relative, data_dir, 'polarization-all-by-tactic-dist-relative-LATEX.tex')
  print('finished breakdown by tactic and distribution')

  return polarization_all_df

def polarization_all_analysis(multidata, slope_threshold, intercept_threshold):
  polarization_data = { key: value for (key,value) in multidata.items() if key[1] == 'polarization' }
  x = np.array([[val] for val in range(len(list(polarization_data.values())[0]['0'][0]))])
  all_df = pd.DataFrame(columns=multidata['params'] + ['run','category','lr-intercept','lr-slope','start','end','delta','max','data'])
  for (props, data) in polarization_data.items():
    for i in range(len(data['0'])):
      run_data = data['0'][i]
      model = LinearRegression().fit(x, run_data)
      category = determine_polarization_categories(run_data, model, slope_threshold, intercept_threshold)
      all_df.loc[len(all_df.index)] = list(props[0]) + [i,category,model.intercept_,model.coef_[0],run_data[0],run_data[-1],run_data[-1]-run_data[0],max(run_data),run_data]
  polarizing = all_df[all_df['category'] == 'polarized']
  depolarizing = all_df[all_df['category'] == 'depolarized']
  remained_polarized = all_df[all_df['category'] == 'remained_polarized']
  remained_nonpolarized = all_df[all_df['category'] == 'remained_nonpolarized']

  return { 'polarization_df': all_df, 'polarized': polarizing, 'depolarized': depolarizing, 'remained_polarized': remained_polarized, 'remained_nonpolarized': remained_nonpolarized }

def polarization_means_analysis(multidata,slope_threshold,intercept_threshold):
  '''
  Analyze polarization data for any of the experiments' multidata
  collection. This returns a data frame with conditions parameters
  and measures on the polarization data like linear regression slope,
  intercept, min, max, final value of the mean values, variance, etc.

  This reports data broken up by a polarization regression slope threshold
  and thus partitions the results into polarizing, depolarizing, and
  remaining the same.

  :param multidata: A collection of multidata for a given experiment.
  :param slope_threshold: A slope value to use to categorize results
  as polarizing or not based off of slope of a linear regression fit
  to their data.
  :param intercept_threshold: A y-value to use to categorize results
  as being polarized or not based off of the intercept of a linear
  regression fit to their data.
  '''
  # slope_threshold = 0.01
  # intercept_threshold = 8.5

  polarization_data = { key: value for (key,value) in multidata.items() if key[1] == 'polarization' }
  polarization_means = { key: value['0'].mean(0) for (key,value) in polarization_data.items() }
  polarization_vars = { key: value['0'].var(0).mean() for (key,value) in polarization_data.items() }
  x = np.array([[val] for val in range(len(list(polarization_means.values())[0]))])
  df = pd.DataFrame(columns=multidata['params'] + ['category','lr-intercept','lr-slope','var','start','end','delta','max','data'])

  for (props, data) in polarization_means.items():
    model = LinearRegression().fit(x, data)
    category = determine_polarization_categories(data, model, slope_threshold, intercept_threshold)
    df.loc[len(df.index)] = list(props[0]) + [category,model.intercept_,model.coef_[0],polarization_vars[props],data[0],data[-1],data[-1]-data[0],max(data),data]

  polarizing = df[df['category'] == 'polarized']
  depolarizing = df[df['category'] == 'depolarized']
  remained_polarized = df[df['category'] == 'remained_polarized']
  remained_nonpolarized = df[df['category'] == 'remained_nonpolarized']

  return { 'polarization_df': df, 'polarized': polarizing, 'depolarized': depolarizing, 'remained_polarized': remained_polarized, 'remained_nonpolarized': remained_nonpolarized }