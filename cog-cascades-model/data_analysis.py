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
import numpy as np
from scipy.stats import chi2_contingency, truncnorm, pearsonr
from sklearn.linear_model import LinearRegression
import math
import matplotlib.pyplot as plt
from nlogo_io import *
from messaging import dist_to_agent_brain, believe_message
# import statsmodels.formula.api as smf

DATA_DIR = 'D:/school/grad-school/Tufts/research/cog-cascades-trust'



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
    if in_filename in file:
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
        elif abs(len(data_vector) - full_data_size) <= 5:
          data_vector = np.append(data_vector, [ data_vector[-1] for i in range(abs(len(data_vector) - full_data_size)) ])
        else:
          print(f'ERROR parsing multi chart data -- data length {len(data_vector)} did not equal number of ticks {full_data_size}')
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

  if list(multi_data.keys())[0] != 'default':
    # This is specific code to set the colors for belief resolutions
    multi_data_keys_int = list(map(lambda el: int(el), multi_data.keys()))
    resolution = int(max(multi_data_keys_int))+1
    line_color = lambda key: f"#{rgb_to_hex([ 255 - round((255/max(resolution-1,1))*int(key)), 0, round((255/max(resolution-1,1)) * int(key)) ])}"
 
  multi_data_has_multiple = lambda multi_data_entry: type(multi_data_entry[0]) == type(np.array(0)) and len(multi_data_entry) > 1
 
  for key in multi_data:
    mean_vec = multi_data[key].mean(0) if multi_data_has_multiple(multi_data[key])  else multi_data[key]
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
    print(f'processing {param_measure}')
    if param_measure == 'params':
      continue
    # Something failed to read/didn't exist
    if data == -1:
      continue
    param_combo = param_measure[0]
    measure = param_measure[1]
    for (pen_name,runs_data) in data.items():
      print(f'processing runs for pen {pen_name}')
      for run in range(len(runs_data)):
        print(f'run {run}')
        df.loc[len(df.index)] = list(param_combo) + [measure,pen_name,run,runs_data[run]]
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

def process_low_res_param_sweep_exp(path):
  cognitive_translate = ['0', '1', '2']
  institution_tactic = ['broadcast-brain', 'appeal-mean']
  media_ecosystem_n = ['15']
  media_ecosystem_dist = [ 'uniform', 'normal', 'polarized' ]
  init_cit_dist = ['normal', 'uniform', 'polarized']
  zeta_media = ['0.25','0.5','0.75']
  zeta_cit = ['0.25','0.5','0.75']
  citizen_memory_length = ['1','2','10']
  repetition = list(map(str, range(5)))
  process_exp_outputs(
    [cognitive_translate,institution_tactic,media_ecosystem_dist,media_ecosystem_n,init_cit_dist,zeta_cit,zeta_media,citizen_memory_length,repetition],
    {'percent-agent-beliefs': [PLOT_TYPES.LINE, PLOT_TYPES.STACK],
    'polarization': [PLOT_TYPES.LINE],
    'fragmentation': [PLOT_TYPES.LINE],
    'homophily': [PLOT_TYPES.LINE]},
    path)

def process_parameter_sweep_test_exp_messages(path):
  # Note: This is a small parameter sapce to save processing
  # time for testing
  cognitive_translate = ['0']
  institution_tactic = ['broadcast-brain']
  media_ecosystem_n = ['20']
  media_ecosystem_dist = [ 'uniform', 'normal']
  init_cit_dist = ['normal', 'uniform']
  zeta_media = ['0.25','0.5','0.75']
  zeta_cit = ['0.25','0.5','0.75']
  citizen_memory_length = ['5']
  repetition = list(map(str, range(2)))
  message_multidata = process_all_message_multidata(
    [cognitive_translate,institution_tactic,media_ecosystem_dist,media_ecosystem_n,init_cit_dist,zeta_media,zeta_cit,citizen_memory_length,repetition],
    path
  )
  return message_multidata

def process_parameter_sweep_test_exp(path):
  cognitive_translate = ['0', '1']
  institution_tactic = ['broadcast-brain', 'appeal-mean']
  media_ecosystem_n = ['20']
  media_ecosystem_dist = [ 'uniform', 'normal', 'polarized' ]
  init_cit_dist = ['normal', 'uniform', 'polarized']
  zeta_media = ['0.25','0.5','0.75','1']
  zeta_cit = ['0.25','0.5','0.75','1']
  citizen_memory_length = ['5']
  ba_m = ['3']
  graph_type = ['barabasi-albert']
  repetition = list(map(str, range(2)))
  process_exp_outputs(
    [cognitive_translate,institution_tactic,media_ecosystem_dist,media_ecosystem_n,init_cit_dist,zeta_media,zeta_cit,citizen_memory_length,repetition],
    {'percent-agent-beliefs': [PLOT_TYPES.LINE, PLOT_TYPES.STACK],
    # 'polarization': [PLOT_TYPES.LINE]},
    'polarization': [PLOT_TYPES.LINE],
    'fragmentation': [PLOT_TYPES.LINE],
    'homophily': [PLOT_TYPES.LINE]},
    path)
  message_multidata = get_all_message_multidata(
    [cognitive_translate,institution_tactic,media_ecosystem_dist,media_ecosystem_n,init_cit_dist,zeta_media,zeta_cit,citizen_memory_length,repetition],
    path
  )
  return message_multidata

def process_parameter_sweep_tinytest_exp(path):
  cognitive_translate = ['0', '1', '2']
  epsilon = ['0']
  institution_tactic = ['broadcast-brain', 'appeal-mean']
  media_ecosystem_n = ['20']
  media_ecosystem_dist = [ 'uniform', 'normal', 'polarized' ]
  init_cit_dist = ['normal', 'uniform', 'polarized']
  citizen_memory_length = ['5']
  repetition = list(map(str, range(2)))

  process_exp_outputs(
    [cognitive_translate,institution_tactic,media_ecosystem_dist,media_ecosystem_n,init_cit_dist,epsilon,citizen_memory_length,repetition],
    {'percent-agent-beliefs': [PLOT_TYPES.LINE, PLOT_TYPES.STACK],
    'polarization': [PLOT_TYPES.LINE]},
    # 'polarization': [PLOT_TYPES.LINE],
    # 'fragmentation': [PLOT_TYPES.LINE],
    # 'homophily': [PLOT_TYPES.LINE]},
    path)

def get_static_param_sweep_high_N_multidata(path):
  cognitive_translate = ['0', '1', '2']
  epsilon = ['0','1','2']
  institution_tactic = ['broadcast-brain', 'appeal-mean']
  media_ecosystem_dist = [ 'uniform', 'normal', 'polarized' ]
  init_cit_dist = ['normal', 'uniform', 'polarized']
  repetition = ['0']
  graph_type = ['barabasi-albert','ba-homophilic']
  ba_m = ['3']
  measure_multidata = get_all_multidata(
    [cognitive_translate,institution_tactic,media_ecosystem_dist,init_cit_dist,epsilon,graph_type,ba_m,repetition],
    ['translate','tactic','media_dist','cit_dist','epsilon','graph_type','ba_m','repetition'],
    {'percent-agent-beliefs': [PLOT_TYPES.LINE, PLOT_TYPES.STACK],
    'polarization': [PLOT_TYPES.LINE]},
    # 'fragmentation': [PLOT_TYPES.LINE],
    # 'homophily': [PLOT_TYPES.LINE]},
    path)
  return measure_multidata

def get_low_res_low_media_multidata(path):
  cognitive_translate = ['0', '1', '2']
  institution_tactic = ['broadcast-brain', 'appeal-mean']
  media_ecosystem_dist = [ 'two-mid', 'two-polarized', 'three-mid', 'three-polarized' ]
  init_cit_dist = ['normal', 'uniform', 'polarized']
  zeta_media = ['0.25','0.5','0.75']
  zeta_cit = ['0.25','0.5','0.75']
  citizen_memory_length = ['1','2','10']
  repetition = list(map(str, range(5)))

  measure_multidata = get_all_multidata(
    [cognitive_translate,media_ecosystem_dist,institution_tactic,init_cit_dist,zeta_cit,zeta_media,citizen_memory_length,repetition],
    ['translate','media_dist','tactic','citizen_dist','zeta_citizen','zeta_media','citizen_memory_len','repetition'],
    {'percent-agent-beliefs': [PLOT_TYPES.LINE, PLOT_TYPES.STACK],
    'polarization': [PLOT_TYPES.LINE],
    'fragmentation': [PLOT_TYPES.LINE],
    'homophily': [PLOT_TYPES.LINE]},
    path)
  return measure_multidata

def get_low_res_sweep_multidata(path):
  cognitive_translate = ['0', '1', '2']
  institution_tactic = ['broadcast-brain', 'appeal-mean']
  media_ecosystem_n = ['15']
  media_ecosystem_dist = [ 'uniform', 'normal', 'polarized' ]
  init_cit_dist = ['normal', 'uniform', 'polarized']
  zeta_media = ['0.25','0.5','0.75']
  zeta_cit = ['0.25','0.5','0.75']
  citizen_memory_length = ['1','2','10']
  repetition = list(map(str, range(5)))

  measure_multidata = get_all_multidata(
    [cognitive_translate,institution_tactic,media_ecosystem_dist,media_ecosystem_n,init_cit_dist,zeta_cit,zeta_media,citizen_memory_length,repetition],
    ['translate','tactic','media_dist','media_n','citizen_dist','zeta_citizen','zeta_media','citizen_memory_len','repetition'],
    {'percent-agent-beliefs': [PLOT_TYPES.LINE, PLOT_TYPES.STACK],
    'polarization': [PLOT_TYPES.LINE],
    'fragmentation': [PLOT_TYPES.LINE],
    'homophily': [PLOT_TYPES.LINE]},
    path)
  return measure_multidata

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

def dynamic_model_total_analysis(data_dir, data_file, params):
  print('loading polarization data...')
  df = read_polarization_dataframe(f'{data_dir}/{data_file}')
  df_no_disconnected = df[(df['zeta_media'] != 1) & (df['zeta_citizen'] != 1) & ((df['zeta_media'] != 0.75) | (df['translate'] != 0)) & ((df['zeta_citizen'] != 0.75) | (df['translate'] != 0))]
  df_no_disconnected = df_no_disconnected.drop(columns=['Unnamed: 0'])
  multidata = dataframe_as_multidata(df_no_disconnected)
  # multidata['params'] = ['translate','tactic','media_dist','media_n','citizen_dist','zeta_citizen','zeta_media','citizen_memory_len','repetition']
  multidata['params'] = params
  print('loaded polarization data')

  print('starting polarization analysis...')
  polarization_slope = 0.01
  polarization_intercept = 5.5

  polarization_categories = ['polarized','depolarized','remained_polarized','remained_nonpolarized']

  polarization_data = polarization_all_analysis(multidata, polarization_slope, polarization_intercept)
  with open(f'{data_dir}/polarization-results.json','w') as f:
    polarization_results = { category: len(polarization_data[category]) for category in polarization_categories }
    json.dump(polarization_results, f)
  print('finished polarization analysis')
 
  # polarization_mean_df = polarization_data['polarization_df']
  polarization_all_df = polarization_data['polarization_df']

  print('starting polarization stability analysis...')
  polarization_stability_data = polarization_stability_analysis(multidata, polarization_slope, polarization_intercept)
  with open(f'{data_dir}/polarization-stability-diff-parts.json','w') as f:
    json.dump(polarization_stability_data['diff_parts'], f)
  print('finished polarization stability analysis')

  # stability_df = polarization_stability_data['stability']

  # print('starting polarization analysis across repetitions...')
  # polarization_data_across_runs = polarization_analysis_across_repetitions(polarization_mean_df, multidata, polarization_slope, polarization_intercept)
  # with open(f'{data_dir}/polarization-across-reps-results.json','w') as f:
  #   polarization_results = { category: len(polarization_data_across_runs[category]) for category in polarization_categories }
  #   json.dump(polarization_results, f)
  # print('finished polarization analysis across repetitions')

  # print('starting polarization stability analysis across repetitions...')
  # polarization_stability_data_across_runs = polarization_stability_across_repetitions(stability_df, multidata)
  # with open(f'{data_dir}/polarization-stability-across-reps-diff-parts.json','w') as f:
  #   json.dump(polarization_stability_data_across_runs['diff_parts'], f)
  # print('finished polarization stability analysis across repetitions')

  print('polarization analyses done')

  # polarization_df_across_runs = polarization_data_across_runs['polarization_df']

  print('starting polarization results by parameter breakdown...')

  (mem_relative, mem_absolute) = dynamic_polarization_results_by_memory_len(polarization_all_df)
  with open(f'{data_dir}/polarization-by-mem-relative.json','w') as f:
    json.dump(mem_relative, f)
  with open(f'{data_dir}/polarization-by-mem-absolute.json','w') as f:
    json.dump(mem_absolute, f)
  write_dynamic_polarization_by_memory_len(mem_absolute, data_dir, 'polarization-by-mem-absolute-LATEX.tex')
  write_dynamic_polarization_by_memory_len(mem_relative, data_dir, 'polarization-by-mem-relative-LATEX.tex')
  print('finished breakdown by memory len')

  (fe_relative, fe_absolute) = dynamic_polarization_results_by_fragmentation_exposure(polarization_all_df)
  with open(f'{data_dir}/polarization-by-frag-zeta-relative.json','w') as f:
    json.dump(fe_relative, f)
  with open(f'{data_dir}/polarization-by-frag-zeta-absolute.json','w') as f:
    json.dump(fe_absolute, f)
  write_dynamic_polarization_by_fragmentation_exposure(fe_absolute, data_dir, 'polarization-by-frag-zeta-absolute-LATEX.tex')
  write_dynamic_polarization_by_fragmentation_exposure(fe_relative, data_dir, 'polarization-by-frag-zeta-relative-LATEX.tex')
  print('finished breakdown by fragmentation and exposure')

  (fe_mem_relative,fe_mem_absolute) = dynamic_polarization_results_by_fragmentation_exposure_memory(polarization_all_df)
  # write_dynamic_polarization_by_fragmentation_exposure_mem(proportions, data_dir, filename)

  # (fe_relative_runs, fe_absolute_runs) = dynamic_polarization_results_by_fragmentation_exposure(polarization_df_across_runs)
  # with open(f'{data_dir}/polarization-across-reps-by-frag-zeta-relative.json','w') as f:
  #   json.dump(fe_relative_runs, f)
  # with open(f'{data_dir}/polarization-across-reps-by-frag-zeta-absolute.json','w') as f:
  #   json.dump(fe_absolute_runs, f)
  # write_dynamic_polarization_by_fragmentation_exposure(fe_absolute_runs, data_dir, 'polarization-across-reps-by-frag-zeta-absolute-LATEX.tex')
  # write_dynamic_polarization_by_fragmentation_exposure(fe_relative_runs, data_dir, 'polarization-across-reps-by-frag-zeta-relative-LATEX.tex')
  # print('finished breakdown by fragmentation and exposure')

  tactic_dist_df = polarization_all_df
  # tactic_dist_df = polarization_all_df[(polarization_all_df['translate'] == 1) | (polarization_all_df['translate'] == 2)]
  (td_relative, td_absolute) = polarization_results_by_tactic_distributions(tactic_dist_df)
  with open(f'{data_dir}/polarization-all-by-tactic-dist-relative.json','w') as f:
    json.dump(td_relative, f)
  with open(f'{data_dir}/polarization-all-by-tactic-dist-absolute.json','w') as f:
    json.dump(td_absolute, f)
  write_polarization_by_tactic_distribution(td_absolute, data_dir, 'polarization-all-by-tactic-dist-absolute-LATEX.tex')
  write_polarization_by_tactic_distribution(td_relative, data_dir, 'polarization-all-by-tactic-dist-relative-LATEX.tex')
  print('finished breakdown by tactic and distribution')

  print('finished polarization results by parameter breakdown')

  print('starting polarization correlation analysis...')
  # fragmentation_data = fragmentation_analysis(multidata)
  # fragmentation_all_df = fragmentation_data['fragmentation_all_df']
  # print('fragmentation data gathered')
  # homophily_data = homophily_analysis(multidata)
  # homophily_all_df = homophily_data['homophily_all_df']
  # print('homophily data gathered')
  # correlation_df = correlation_polarization_fragmentation_homophily_all(polarization_all_df, fragmentation_all_df, homophily_all_df, multidata)
  # correlation_df.to_csv(f'{data_dir}/correlation-pol-frag-homo-all.csv')
  # polarized_correlations = correlation_values_for_polarized(polarization_all_df, correlation_df)
  # polarized_correlations.to_csv(f'{data_dir}/polarized-correlation-pol-frag-homo-all.csv')

  # Old correlation analysis
  # pol_frag_corr = correlation_polarization_fragmentation_means(polarization_all_df, fragmentation_all_df, multidata, f'{data_dir}/polarization-fragmentation-correlation.png')
  # with open(f'{data_dir}/polarization-fragmentation-correlation.txt','w') as f:
  #   f.write(str(pol_frag_corr))
  print('finished polarization correlation analysis')

  # pol_frag_corr = correlation_polarization_homophily_means(polarization_all_df, homophily_all_df, multidata, f'{data_dir}/polarization-homophily-correlation.png')
  # with open(f'{data_dir}/polarization-homophily-correlation.txt','w') as f:
  #   f.write(str(pol_frag_corr))

  return (polarization_all_df, fragmentation_all_df, homophily_all_df, correlation_df)

def runs_with_unconnected_institution_graphs(graphs_path):
  '''
  Analyze the files in the graphs directory to find which ones
  initially have citizen agents connected to media agents, and which ones
  are unconnected.

  :param graphs_path: The path to the graph output folder from the
  BehaviorSpace simulation run.
  '''
  connected_graphs = []
  unconnected_graphs = {}
  df = pd.DataFrame(columns=['translate','tactic','media_dist','media_n','citizen_dist','zeta_citizen','zeta_media','citizen_memory_len','repetition','num_disconnected','media_without_edges'])
  if os.path.isdir(graphs_path):
    for file in os.listdir(graphs_path):
      citizens, cit_social_edges, media, media_sub_edges  = read_graph(f'{graphs_path}/{file}')
      media_with_edges = set([ m[1] for m in media_sub_edges ])
      media_set = set([ m[0] for m in media ])
      media_without_edges = media_set.difference(media_with_edges)
      param_combo = file.replace('.csv','').replace('appeal-mean','appeal_mean').replace('broadcast-brain','broadcast_brain').split('-')
      # print(param_combo)
      # This is hardcoded to be half of the 15 we simulated
      df.loc[len(df.index)] = param_combo + [len(media_without_edges),media_without_edges]
  return df

def missing_values_in_data(multidata):
  missing_vals_polarization = { key: len(val['0']) for (key,val) in multidata.items() if key[1]=='polarization' and len(val['0']) != 5 }
  missing_vals_fragmentation = { key: len(val['default']) for (key,val) in multidata.items() if key[1]=='fragmentation' and len(val['default']) != 5 }
  missing_vals_homophily = { key: len(val['default']) for (key,val) in multidata.items() if key[1]=='homophily' and len(val['default']) != 5 }

  missing_values = {
    'polarization': { ','.join(key[0]): 5-val for key,val in missing_vals_polarization.items() },
    'fragmentation': { ','.join(key[0]): 5-val for key,val in missing_vals_fragmentation.items() },
    'homophily': { ','.join(key[0]): 5-val for key,val in missing_vals_homophily.items() }
  }
  return missing_values

def logistic_regression_polarization(polarization_data):
  '''
  Run a logistic regression to fit polarization data given different
  combinations of simulation parameters.

  This analysis supports results in Rabb & Cowen, 2022 in Section 5
  where we discuss the effect of parameters on polarization results
  reported in Tables 3-5.

  :param polarization_data: The result of polarization_analysis(multidata)
  This contains 2 key dataframes -- one for polarizing results, one for
  nonpolarizing ones
  '''
  polarizing = polarization_data['polarizing']
  nonpolarizing = polarization_data['nonpolarizing']
  polarizing['polarized'] = 1
  nonpolarizing['polarized'] = 0
  
  df = polarizing.append(nonpolarizing)
  df['epsilon'] = df['epsilon'].astype("int64")
  df['translate'] = df['translate'].astype("int64")
  df['graph_type'] = df['graph_type'].astype("category")
  df['tactic'] = df['tactic'].astype("category")
  df['citizen_dist'] = df['citizen_dist'].astype("category")
  df['media_dist'] = df['media_dist'].astype("category")

  # This model yields results discussed in Subsection 5.1, the effect
  # of h_G, epsilon, and gamma on polarization results.
  result = smf.logit("polarized ~ epsilon + translate + graph_type", data=df).fit()
  print(result.summary())

  # This model yields results discussed in Subsection 5.2, Table 5,
  # the effect of C, I and gamma on polarization results. To select
  # I = 'uniform' or I='polarized', different lines can be commented
  # or uncommented.
  df = df[df['tactic']=='broadcast-brain']
  # df = df[df['media_dist']=='uniform']
  df = df[df['media_dist']=='polarized']
  result = smf.logit("polarized ~ epsilon + translate + graph_type + citizen_dist", data=df).fit()

  print(result.summary())

def polarization_results_by_tactic_exposure(polarization_data):
  '''
  Run an analysis to see how many results polarized vs nonpolarized for
  parameter combinations of varphi (tactic) and gamma (translate).

  This analysis supports Table 4 in Rabb & Cowen 2022.
  
  :param polarization_data: The result of polarization_analysis(multidata)
  This contains 2 key dataframes -- one for polarizing results, one for
  nonpolarizing ones
  '''
  polarizing = polarization_data['polarizing']
  nonpolarizing = polarization_data['nonpolarizing']
  polarizing['polarized'] = 1
  nonpolarizing['polarized'] = 0
  all_results = polarizing.append(nonpolarizing)

  dfs = {
    "all_broadcast": all_results[all_results['tactic'] == 'broadcast-brain'],
    "all_mean": all_results[all_results['tactic'] == 'appeal-mean'],
    "all_median": all_results[all_results['tactic'] == 'appeal-median'],
  }

  gamma_values = [0,1,2]
  proportions = {}
  for (df_name, df) in dfs.items():
    print(f'{df_name}\n==========')
    for gamma in gamma_values:
      partition_polarized = df.query(f'translate=="{gamma}" and polarized==1')
      partition_nonpolarized = df.query(f'translate=="{gamma}" and polarized==0')
      partition_all = df.query(f'translate=="{gamma}"')
      
      # Use this line to report percent of results that are polarized
      proportions[(gamma)] = {'polarized': len(partition_polarized) / len(partition_all), 'nonpolarized': len(partition_nonpolarized) / len(partition_all) }
      
      # Use this line to report number of results that are polarized
      # proportions[(gamma)] = {'polarized': len(partition_polarized), 'nonpolarized': len(partition_nonpolarized) }

def write_static_polarization_by_fragmentation_exposure(proportions, data_dir, filename):
  latex_format = """\\begingroup
    \\setlength{\\tabcolsep}{6pt}
    \\renewcommand{\\arraystretch}{1.5}
    \\begin{table}[]
      \\centering
      \\begin{tabular}{c||c|c|c||c|c|c}
      $h_G$ &\\multicolumn{3}{c||}{1}&\\multicolumn{3}{c||}{$h(b_u,b_v)$}\\\\
      \\hline
      \\hline
      $\\epsilon=2$ & 2,barabasi-albert,0 & 2,barabasi-albert,1 & 2,barabasi-albert,2 & 2,ba-homophilic,0 & 2,ba-homophilic,1 & 2,ba-homophilic,2\\\\
      \\hline
      $\\epsilon=1$ & 1,barabasi-albert,0 & 1,barabasi-albert,1 & 1,barabasi-albert,2 & 1,ba-homophilic,0 & 1,ba-homophilic,1 & 1,ba-homophilic,2\\\\
      \\hline
      $\\epsilon=0$ & 0,barabasi-albert,0 & 0,barabasi-albert,1 & 0,barabasi-albert,2 & 0,ba-homophilic,0 & 0,ba-homophilic,1 & 0,ba-homophilic,2\\\\
      \\hline
      \\hline
      $\\gamma$ & 0 & 1 & 2 & 0 & 1 & 2\\\\
      \\end{tabular}
      \\caption{Percentage of polarized / nonpolarized results (over the 36 experiments in each cell) broken down by selective exposure ($\gamma$ and $h_G$) and fragmentation ($\epsilon$).}
      \\label{tab:results-epsilon-gamma-hg}
    \\end{table}
    \\endgroup"""
  latex_format_four_cats = latex_format
  latex_format_two_cats = latex_format
  for (key,val) in proportions.items():
    if key == 'key': continue
    polarized = val['polarized']
    depolarized = val['depolarized']
    remained_polarized = val['remained_polarized']
    remained_nonpolarized = val['remained_nonpolarized']

    key_pieces = key.split(',')
    zeta_c = key_pieces[0]
    zeta_i = key_pieces[1]
    translate = key_pieces[2]

    latex_format_four_cats = latex_format_four_cats.replace(f'{zeta_c},{zeta_i},{translate}', f'{polarized}/{depolarized}/{remained_polarized}/{remained_nonpolarized}')
    latex_format_two_cats = latex_format_two_cats.replace(f'{zeta_c},{zeta_i},{translate}', f'{polarized+remained_polarized}/{depolarized+remained_nonpolarized}')

  with open(f'{data_dir}/{filename}'.replace('.tex','_all-categories.tex'),'w') as f:
    f.write(latex_format_four_cats)
  with open(f'{data_dir}/{filename}'.replace('.tex','_two-categories.tex'),'w') as f:
    f.write(latex_format_two_cats)

def write_dynamic_polarization_by_fragmentation_exposure_mem(proportions, data_dir, filename):
  mem_lens = set([ key.split(',')[0] for key in proportions.keys() ])
  by_memory = {}
  for mem_len in mem_lens:
    if mem_len not in by_memory and mem_len != 'key': by_memory[mem_len] = {}
    by_memory[mem_len] = { ','.join(key.split(',')[1:]): val for key,val in proportions.items() if key.split(',')[0] == mem_len }
  by_memory['key'] = ','.join(proportions['key'].split(',')[1:])
  for mem_len,proportion in by_memory.items():
    if mem_len == 'key': continue
    write_dynamic_polarization_by_fragmentation_exposure(proportion, data_dir, f'{filename.replace(".tex", "_mem-" + mem_len + ".tex")}')
  return by_memory

def write_dynamic_polarization_by_fragmentation_exposure(proportions, data_dir, filename):
  latex_format = """\\begingroup
    \\setlength{\\tabcolsep}{6pt}
    \\renewcommand{\\arraystretch}{1.5}
    \\begin{table}[]
      \\centering
      \\begin{tabular}{c||c|c|c||c|c|c||c|c|c}
      $\\zeta_i$ &\\multicolumn{3}{c||}{0.25}&\\multicolumn{3}{c||}{0.5}&\\multicolumn{3}{c}{0.75}\\\\
      \\hline
      \\hline
      $\\gamma=2$ & 0.25,0.25,2 & 0.25,0.5,2 & 0.25,0.75,2 & 0.5,0.25,2 & 0.5,0.5,2 & 0.5,0.75,2 & 0.75,0.25,2 & 0.75,0.5,2 & 0.75,0.75,2\\\\
      \\hline
      $\\gamma=1$ & 0.25,0.25,1 & 0.25,0.5,1 & 0.25,0.75,1 & 0.5,0.25,1 & 0.5,0.5,1 & 0.5,0.75,1 & 0.75,0.25,1 & 0.75,0.5,1 & 0.75,0.75,1\\\\
      \\hline
      $\\gamma=0$ & 0.25,0.25,0 & 0.25,0.5,0 & 0.25,0.75,0 & 0.5,0.25,0 & 0.5,0.5,0 & 0.5,0.75,0 & 0.75,0.25,0 & 0.75,0.5,0 & 0.75,0.75,0\\\\
      \\hline
      \\hline
      $\\zeta_c$ & 0.25 & 0.5 & 0.75 & 0.25 & 0.5 & 0.75 & 0.25 & 0.5 & 0.75\\
      \\end{tabular}
      \\caption{Percentage of polarized / nonpolarized results (over the 36 experiments in each cell) broken down by selective exposure ($\gamma$ and $h_G$) and fragmentation ($\epsilon$).}
      \\label{tab:results-epsilon-gamma-hg}
    \\end{table}
    \\endgroup"""
  latex_format_four_cats = latex_format
  latex_format_two_cats = latex_format
  for (key,val) in proportions.items():
    if key == 'key': continue
    polarized = val['polarized']
    depolarized = val['depolarized']
    remained_polarized = val['remained_polarized']
    remained_nonpolarized = val['remained_nonpolarized']

    key_pieces = key.split(',')
    zeta_c = key_pieces[0]
    zeta_i = key_pieces[1]
    translate = key_pieces[2]

    latex_format_four_cats = latex_format_four_cats.replace(f'{zeta_i},{zeta_c},{translate}', f'{polarized}/{depolarized}/{remained_polarized}/{remained_nonpolarized}')
    latex_format_two_cats = latex_format_two_cats.replace(f'{zeta_i},{zeta_c},{translate}', f'{polarized+remained_polarized}/{depolarized+remained_nonpolarized}')

  with open(f'{data_dir}/{filename}'.replace('.tex','_all-categories.tex'),'w') as f:
    f.write(latex_format_four_cats)
  with open(f'{data_dir}/{filename}'.replace('.tex','_two-categories.tex'),'w') as f:
    f.write(latex_format_two_cats)

def write_dynamic_polarization_by_memory_len(proportions, data_dir, filename):
  latex_format = """\\begingroup
    \\setlength{\\tabcolsep}{6pt}
    \\renewcommand{\\arraystretch}{1.5}
    \\begin{table}[]
      \\centering
      \\begin{tabular}{c|c|c}
      $r=1$ & $r=2$ & $r=10$\\\\
      \\hline
      \\hline
      <<1>> & <<2>> & <<10>>\\\\
      \\end{tabular}
      \\caption{Percentage of polarized / nonpolarized results (over the 36 experiments in each cell) broken down by selective exposure ($\gamma$ and $h_G$) and fragmentation ($\epsilon$).}
      \\label{tab:results-epsilon-gamma-hg}
    \\end{table}
    \\endgroup"""
  latex_format_four_cats = latex_format
  latex_format_two_cats = latex_format
  for (key,val) in proportions.items():
    if key == 'key': continue
    polarized = val['polarized']
    depolarized = val['depolarized']
    remained_polarized = val['remained_polarized']
    remained_nonpolarized = val['remained_nonpolarized']

    key_pieces = key.split(',')
    mem_len = key_pieces[0]

    latex_format_four_cats = latex_format_four_cats.replace(f'<<{mem_len}>>', f'{polarized}/{depolarized}/{remained_polarized}/{remained_nonpolarized}')
    latex_format_two_cats = latex_format_two_cats.replace(f'<<{mem_len}>>', f'{polarized+remained_polarized}/{depolarized+remained_nonpolarized}')

  with open(f'{data_dir}/{filename}'.replace('.tex','_all-categories.tex'),'w') as f:
    f.write(latex_format_four_cats)
  with open(f'{data_dir}/{filename}'.replace('.tex','_two-categories.tex'),'w') as f:
    f.write(latex_format_two_cats)

def dynamic_polarization_results_by_memory_len(df):
  '''
  Run an analysis to see how many results polarized vs nonpolarized for
  parameter combinations of h_G (homophily), epislon, and gamma (translate).

  This analysis supports Table 3 in Rabb & Cowen 2022.
  
  :param polarization_data: The result of polarization_analysis(multidata)
  This contains 2 key dataframes -- one for polarizing results, one for
  nonpolarizing ones
  '''
  # df = polarization_data['polarization_df']
  cit_memory_lens = [1,2,10]
  relative_proportions = {}
  absolute_proportions = {}
  for memory_len in cit_memory_lens:
    partition_polarized = df.query(f'citizen_memory_len=={memory_len} and category=="polarized"')
    partition_depolarized = df.query(f'citizen_memory_len=={memory_len} and category=="depolarized"')
    partition_remained_polarized = df.query(f'citizen_memory_len=={memory_len} and category=="remained_polarized"')
    partition_remained_nonpolarized = df.query(f'citizen_memory_len=={memory_len} and category=="remained_nonpolarized"')
    partition_all = df.query(f'citizen_memory_len=={memory_len}')
    if len(partition_all) == 0:
      partition_all = pd.DataFrame({'empty': [1]})

    # Use this line to report percent of results that are polarized
    relative_proportions[f'{memory_len}'] = {'polarized': round(100 * (len(partition_polarized) / len(partition_all))), 'depolarized': round(100 * (len(partition_depolarized) / len(partition_all))), 'remained_polarized': round(100 * (len(partition_remained_polarized) / len(partition_all))), 'remained_nonpolarized': round(100 * (len(partition_remained_nonpolarized) / len(partition_all))) }

    # Use this line to report number of results that are polarized
    absolute_proportions[f'{memory_len}'] = {'polarized': len(partition_polarized) , 'depolarized': len(partition_depolarized) , 'remained_polarized': len(partition_remained_polarized) , 'remained_nonpolarized': len(partition_remained_nonpolarized), 'total': len(partition_all) }
  relative_proportions['key'] = 'memory_len'
  absolute_proportions['key'] = 'memory_len'
  return (relative_proportions, absolute_proportions)

def dynamic_polarization_results_by_fragmentation_exposure(df):
  '''
  Run an analysis to see how many results polarized vs nonpolarized for
  parameter combinations of h_G (homophily), epislon, and gamma (translate).

  This analysis supports Table 3 in Rabb & Cowen 2022.
  
  :param polarization_data: The result of polarization_analysis(multidata)
  This contains 2 key dataframes -- one for polarizing results, one for
  nonpolarizing ones
  '''
  # df = polarization_data['polarization_df']
  zeta_cit_values = [0.25,0.5,0.75]
  zeta_media_values = [0.25,0.5,0.75]
  gamma_values = [0,1,2]
  relative_proportions = {}
  absolute_proportions = {}
  for zeta_cit in zeta_cit_values:
    for zeta_media in zeta_media_values:
      for gamma in gamma_values:
        partition_polarized = df.query(f'zeta_citizen=={zeta_cit} and zeta_media=={zeta_media} and  translate=={gamma} and category=="polarized"')
        partition_depolarized = df.query(f'zeta_citizen=={zeta_cit} and zeta_media=={zeta_media} and translate=={gamma} and category=="depolarized"')
        partition_remained_polarized = df.query(f'zeta_citizen=={zeta_cit} and zeta_media=={zeta_media} and translate=={gamma} and category=="remained_polarized"')
        partition_remained_nonpolarized = df.query(f'zeta_citizen=={zeta_cit} and zeta_media=={zeta_media} and translate=={gamma} and category=="remained_nonpolarized"')
        partition_all = df.query(f'zeta_citizen=={zeta_cit} and zeta_media=={zeta_media} and translate=={gamma}')
        if len(partition_all) == 0:
          partition_all = pd.DataFrame({'empty': [1]})

        # Use this line to report percent of results that are polarized
        relative_proportions[f'{zeta_cit},{zeta_media},{gamma}'] = {'polarized': round(100 * (len(partition_polarized) / len(partition_all))), 'depolarized': round(100 * (len(partition_depolarized) / len(partition_all))), 'remained_polarized': round(100 * (len(partition_remained_polarized) / len(partition_all))), 'remained_nonpolarized': round(100 * (len(partition_remained_nonpolarized) / len(partition_all))) }

        # Use this line to report number of results that are polarized
        absolute_proportions[f'{zeta_cit},{zeta_media},{gamma}'] = {'polarized': len(partition_polarized) , 'depolarized': len(partition_depolarized) , 'remained_polarized': len(partition_remained_polarized) , 'remained_nonpolarized': len(partition_remained_nonpolarized), 'total': len(partition_all) }
  relative_proportions['key'] = 'zeta_cit,zeta_media,translate'
  absolute_proportions['key'] = 'zeta_cit,zeta_media,translate'
  return (relative_proportions, absolute_proportions)

def dynamic_polarization_results_by_fragmentation_exposure_memory(df):
  '''
  Run an analysis to see how many results polarized vs nonpolarized for
  parameter combinations of h_G (homophily), epislon, and gamma (translate).

  This analysis supports Table 3 in Rabb & Cowen 2022.
  
  :param polarization_data: The result of polarization_analysis(multidata)
  This contains 2 key dataframes -- one for polarizing results, one for
  nonpolarizing ones
  '''
  # df = polarization_data['polarization_df']
  zeta_cit_values = [0.25,0.5,0.75]
  zeta_media_values = [0.25,0.5,0.75]
  gamma_values = [1,2]
  cit_memory_values = [1,2,10]
  relative_proportions = {}
  absolute_proportions = {}
  for cit_memory in cit_memory_values:
    for zeta_cit in zeta_cit_values:
      for zeta_media in zeta_media_values:
        for gamma in gamma_values:
          partition_polarized = df.query(f'citizen_memory_len=={cit_memory} and zeta_citizen=={zeta_cit} and zeta_media=={zeta_media} and  translate=={gamma} and category=="polarized"')
          partition_depolarized = df.query(f'citizen_memory_len=={cit_memory} and zeta_citizen=={zeta_cit} and zeta_media=={zeta_media} and translate=={gamma} and category=="depolarized"')
          partition_remained_polarized = df.query(f'citizen_memory_len=={cit_memory} and zeta_citizen=={zeta_cit} and zeta_media=={zeta_media} and translate=={gamma} and category=="remained_polarized"')
          partition_remained_nonpolarized = df.query(f'citizen_memory_len=={cit_memory} and zeta_citizen=={zeta_cit} and zeta_media=={zeta_media} and translate=={gamma} and category=="remained_nonpolarized"')
          partition_all = df.query(f'citizen_memory_len=={cit_memory} and zeta_citizen=={zeta_cit} and zeta_media=={zeta_media} and translate=={gamma}')
          if len(partition_all) == 0:
            partition_all = pd.DataFrame({'empty': [1]})

          # Use this line to report percent of results that are polarized
          relative_proportions[f'{cit_memory},{zeta_cit},{zeta_media},{gamma}'] = {'polarized': len(partition_polarized) / len(partition_all), 'depolarized': len(partition_depolarized) / len(partition_all), 'remained_polarized': len(partition_remained_polarized) / len(partition_all), 'remained_nonpolarized': len(partition_remained_nonpolarized) / len(partition_all) }

          # Use this line to report number of results that are polarized
          absolute_proportions[f'{cit_memory},{zeta_cit},{zeta_media},{gamma}'] = {'polarized': len(partition_polarized) , 'depolarized': len(partition_depolarized) , 'remained_polarized': len(partition_remained_polarized) , 'remained_nonpolarized': len(partition_remained_nonpolarized), 'total': len(partition_all) }
  relative_proportions['key'] = 'citizen_memory_len,zeta_cit,zeta_media,translate'
  absolute_proportions['key'] = 'citizen_memory_len,zeta_cit,zeta_media,translate'
  return (relative_proportions, absolute_proportions)

def static_polarization_results_by_fragmentation_exposure(df):
  '''
  Run an analysis to see how many results polarized vs nonpolarized for
  parameter combinations of h_G (homophily), epislon, and gamma (translate).

  This analysis supports Table 3 in Rabb & Cowen 2022.
  
  :param polarization_data: The result of polarization_analysis(multidata)
  This contains 2 key dataframes -- one for polarizing results, one for
  nonpolarizing ones
  '''
  # df = polarization_data['polarization_df']
  epsilon_values = [0,1,2]
  graph_types = ['barabasi-albert','ba-homophilic']
  gamma_values = [0,1,2]
  relative_proportions = {}
  absolute_proportions = {}
  for epsilon in epsilon_values:
    for gamma in gamma_values:
      for graph_type in graph_types:
        partition_polarized = df.query(f'epsilon=={epsilon} and graph_type=="{graph_type}" and  translate=={gamma} and category=="polarized"')
        partition_depolarized = df.query(f'epsilon=={epsilon} and graph_type=="{graph_type}" and translate=={gamma} and category=="depolarized"')
        partition_remained_polarized = df.query(f'epsilon=={epsilon} and graph_type=="{graph_type}" and translate=={gamma} and category=="remained_polarized"')
        partition_remained_nonpolarized = df.query(f'epsilon=={epsilon} and graph_type=="{graph_type}" and translate=={gamma} and category=="remained_nonpolarized"')
        partition_all = df.query(f'epsilon=={epsilon} and graph_type=="{graph_type}" and translate=={gamma}')
        if len(partition_all) == 0:
          partition_all = pd.DataFrame({'empty': [1]})

        # Use this line to report percent of results that are polarized
        relative_proportions[f'{epsilon},{graph_type},{gamma}'] = {'polarized': round(100 * (len(partition_polarized) / len(partition_all))), 'depolarized': round(100 * (len(partition_depolarized) / len(partition_all))), 'remained_polarized': round(100 * (len(partition_remained_polarized) / len(partition_all))), 'remained_nonpolarized': round(100 * (len(partition_remained_nonpolarized) / len(partition_all))) }

        # Use this line to report number of results that are polarized
        absolute_proportions[f'{epsilon},{graph_type},{gamma}'] = {'polarized': len(partition_polarized) , 'depolarized': len(partition_depolarized) , 'remained_polarized': len(partition_remained_polarized) , 'remained_nonpolarized': len(partition_remained_nonpolarized), 'total': len(partition_all) }
  relative_proportions['key'] = 'epsilon,graph_type,translate'
  absolute_proportions['key'] = 'epsilon,graph_type,translate'
  return (relative_proportions, absolute_proportions)

def polarization_results_by_tactic_distributions(df):
  '''
  Run an analysis to see how many results polarized vs nonpolarized for
  parameter gamma (translate), citizen distribution and institution
  distribution.
  
  :param polarization_data: The result of polarization_analysis(multidata)
  This contains 2 key dataframes -- one for polarizing results, one for
  nonpolarizing ones
  '''
  tactics = ['broadcast-brain','appeal-mean']
  dist_values = ['uniform','normal','polarized']
  relative_proportions = {}
  absolute_proportions = {}
  for tactic in tactics:
    for cit_dist in dist_values:
      for media_dist in dist_values:
        partition_polarized = df.query(f'tactic=="{tactic}" and citizen_dist=="{cit_dist}" and media_dist=="{media_dist}" and category=="polarized"')
        partition_depolarized = df.query(f'tactic=="{tactic}" and citizen_dist=="{cit_dist}" and media_dist=="{media_dist}" and category=="depolarized"')
        partition_remained_polarized = df.query(f'tactic=="{tactic}" and citizen_dist=="{cit_dist}" and media_dist=="{media_dist}" and category=="remained_polarized"')
        partition_remained_nonpolarized = df.query(f'tactic=="{tactic}" and citizen_dist=="{cit_dist}" and media_dist=="{media_dist}" and category=="remained_nonpolarized"')

        partition_all = df.query(f'tactic=="{tactic}" and citizen_dist=="{cit_dist}" and media_dist=="{media_dist}"')
        if len(partition_all) == 0:
          partition_all = pd.DataFrame({'empty': [1]})

        # Use this line to report percent of results that are polarized
        relative_proportions[f'{tactic},{cit_dist},{media_dist}'] = {'polarized': round(100 * (len(partition_polarized) / len(partition_all))), 'depolarized': round(100 * (len(partition_depolarized) / len(partition_all))), 'remained_polarized': round(100 * (len(partition_remained_polarized) / len(partition_all))), 'remained_nonpolarized': round(100 * (len(partition_remained_nonpolarized) / len(partition_all))) }

        # Use this line to report number of results that are polarized
        absolute_proportions[f'{tactic},{cit_dist},{media_dist}'] = {'polarized': len(partition_polarized) , 'depolarized': len(partition_depolarized) , 'remained_polarized': len(partition_remained_polarized) , 'remained_nonpolarized': len(partition_remained_nonpolarized), 'total': len(partition_all) }
  relative_proportions['key'] = 'tactic,citizen_dist,media_dist'
  absolute_proportions['key'] = 'tactic,citizen_dist,media_dist'
  return (relative_proportions, absolute_proportions)

def polarization_results_by_tactic_distributions_predetermined_media(df):
  '''
  Run an analysis to see how many results polarized vs nonpolarized for
  parameter gamma (translate), citizen distribution and institution
  distribution.
  
  :param polarization_data: The result of polarization_analysis(multidata)
  This contains 2 key dataframes -- one for polarizing results, one for
  nonpolarizing ones
  '''
  tactics = ['broadcast-brain','appeal-mean']
  cit_dist_values = ['uniform','normal','polarized']
  media_dist_values = ['two-mid','two-polarized','three-mid','three-polarized']
  relative_proportions = {}
  absolute_proportions = {}
  for tactic in tactics:
    for cit_dist in cit_dist_values:
      for media_dist in media_dist_values:
        partition_polarized = df.query(f'tactic=="{tactic}" and citizen_dist=="{cit_dist}" and media_dist=="{media_dist}" and category=="polarized"')
        partition_depolarized = df.query(f'tactic=="{tactic}" and citizen_dist=="{cit_dist}" and media_dist=="{media_dist}" and category=="depolarized"')
        partition_remained_polarized = df.query(f'tactic=="{tactic}" and citizen_dist=="{cit_dist}" and media_dist=="{media_dist}" and category=="remained_polarized"')
        partition_remained_nonpolarized = df.query(f'tactic=="{tactic}" and citizen_dist=="{cit_dist}" and media_dist=="{media_dist}" and category=="remained_nonpolarized"')

        partition_all = df.query(f'tactic=="{tactic}" and citizen_dist=="{cit_dist}" and media_dist=="{media_dist}"')
        if len(partition_all) == 0:
          partition_all = pd.DataFrame({'empty': [1]})

        # Use this line to report percent of results that are polarized
        relative_proportions[f'{tactic},{cit_dist},{media_dist}'] = {'polarized': round(100*(len(partition_polarized) / len(partition_all))), 'depolarized': round(100*(len(partition_depolarized) / len(partition_all))), 'remained_polarized': round(100*(len(partition_remained_polarized) / len(partition_all))), 'remained_nonpolarized': round(100*(len(partition_remained_nonpolarized) / len(partition_all))) }

        # Use this line to report number of results that are polarized
        absolute_proportions[f'{tactic},{cit_dist},{media_dist}'] = {'polarized': len(partition_polarized) , 'depolarized': len(partition_depolarized) , 'remained_polarized': len(partition_remained_polarized) , 'remained_nonpolarized': len(partition_remained_nonpolarized), 'total': len(partition_all) }
  relative_proportions['key'] = 'tactic,citizen_dist,media_dist'
  absolute_proportions['key'] = 'tactic,citizen_dist,media_dist'
  return (relative_proportions, absolute_proportions)

def write_polarization_by_tactic_distribution(proportions, data_dir, filename):
  latex_format = """\\begingroup
    \\setlength{\\tabcolsep}{6pt}
    \\renewcommand{\\arraystretch}{1.5}
    \\begin{table}[]
      \\centering
      \\begin{tabular}{c||c|c|c||c|c|c||c|c|c}
      $\\mathcal{I}$ &\\multicolumn{3}{c||}{$\\mathcal{U}$}&\\multicolumn{3}{c||}{$\\mathcal{N}$}&\\multicolumn{3}{c}{$\\mathcal{P}$}\\\\
      \\hline
      \\hline
      $\\varphi=broadcast$ & broadcast-brain,uniform,uniform & broadcast-brain,uniform,normal & broadcast-brain,uniform,polarized & broadcast-brain,normal,uniform & broadcast-brain,normal,normal & broadcast-brain,normal,polarized & broadcast-brain,polarized,uniform & broadcast-brain,polarized,normal & broadcast-brain,polarized,polarized \\\\
      \\hline
      $\\varphi=appeal mean$ & appeal-mean,uniform,uniform & appeal-mean,uniform,normal & appeal-mean,uniform,polarized & appeal-mean,normal,uniform & appeal-mean,normal,normal & appeal-mean,normal,polarized & appeal-mean,polarized,uniform & appeal-mean,polarized,normal & appeal-mean,polarized,polarized \\\\
      \\hline
      $\\mathcal{C}$ & $\\mathcal{U}$ & $\\mathcal{N}$ & $\\mathcal{P}$ & $\\mathcal{U}$ & $\\mathcal{N}$ & $\\mathcal{P}$ & $\\mathcal{U}$ & $\\mathcal{N}$ & $\\mathcal{P}$\\
      \\end{tabular}
      \\caption{Percentage of polarized / depolarized / remained polarized / remained nonpolarized results broken down by media tactic ($\\varphi$) and initial belief distributions ($\\mathcal{C}$ and $\\mathcal{I}$).}
      \\label{tab:results-tactic-distribution}
    \\end{table}
    \\endgroup"""
  latex_format_four_cats = latex_format
  latex_format_two_cats = latex_format
  for (key,val) in proportions.items():
    if key == 'key': continue
    polarized = val['polarized']
    depolarized = val['depolarized']
    remained_polarized = val['remained_polarized']
    remained_nonpolarized = val['remained_nonpolarized']

    key_pieces = key.split(',')
    tactic = key_pieces[0]
    cit_dist = key_pieces[1]
    media_dist = key_pieces[2]

    latex_format_four_cats = latex_format_four_cats.replace(f'{tactic},{media_dist},{cit_dist}', f'{polarized}/{depolarized}/{remained_polarized}/{remained_nonpolarized}')
    latex_format_two_cats = latex_format_two_cats.replace(f'{tactic},{media_dist},{cit_dist}', f'{polarized+remained_polarized}/{depolarized+remained_nonpolarized}')

  with open(f'{data_dir}/{filename}'.replace('.tex','_all-categories.tex'),'w') as f:
    f.write(latex_format_four_cats)
  with open(f'{data_dir}/{filename}'.replace('.tex','_two-categories.tex'),'w') as f:
    f.write(latex_format_two_cats)

def write_polarization_by_tactic_distribution_predetermined_media(proportions, data_dir, filename):
  latex_format = """\\begingroup
    \\setlength{\\tabcolsep}{6pt}
    \\renewcommand{\\arraystretch}{1.5}
    \\begin{table}[]
      \\centering
      \\begin{tabular}{c||c|c|c||c|c|c}
      $\\mathcal{I}$ &\\multicolumn{3}{c||}{$\\mathcal{N}(2)$}&\\multicolumn{3}{c||}{$\\mathcal{P}(2)$}\\\\
      \\hline
      \\hline
      $\\varphi=broadcast$ & broadcast-brain,two-mid,uniform & broadcast-brain,two-mid,normal & broadcast-brain,two-mid,polarized & broadcast-brain,two-polarized,uniform & broadcast-brain,two-polarized,normal & broadcast-brain,two-polarized,polarized & \\\\
      \\hline
      $\\varphi=mean$ & appeal-mean,two-mid,uniform & appeal-mean,two-mid,normal & appeal-mean,two-mid,polarized & appeal-mean,two-polarized,uniform & appeal-mean,two-polarized,normal & appeal-mean,two-polarized,polarized \\\\
      $\\mathcal{I}$ &\\multicolumn{3}{c}{$\\mathcal{N}(3)$}&\\multicolumn{3}{c}{$\\mathcal{P}(3)$}\\\\
      \\hline
      \\hline
      $\\varphi=broadcast$ & broadcast-brain,three-mid,uniform & broadcast-brain,three-mid,normal & broadcast-brain,three-mid,polarized & broadcast-brain,three-polarized,uniform & broadcast-brain,three-polarized,normal & broadcast-brain,three-polarized,polarized
      \\hline
      $\\varphi=mean$ & appeal-mean,three-mid,uniform & appeal-mean,three-mid,normal & appeal-mean,three-mid,polarized & appeal-mean,three-polarized,uniform & appeal-mean,three-polarized,normal & appeal-mean,three-polarized,polarized\\\\
      \\hline
      $\\mathcal{C}$ & $\\mathcal{U}$ & $\\mathcal{N}$ & $\\mathcal{P}$ & $\\mathcal{U}$ & $\\mathcal{N}$ & $\\mathcal{P}$\\\\ 
      \\end{tabular}
      \\caption{Percentage of polarized / depolarized / remained polarized / remained nonpolarized results broken down by media tactic ($\\varphi$) and initial belief distributions ($\\mathcal{C}$ and $\\mathcal{I}$).}
      \\label{tab:results-tactic-distribution}
    \\end{table}
    \\endgroup"""
  latex_format_four_cats = latex_format
  latex_format_two_cats = latex_format
  for (key,val) in proportions.items():
    if key == 'key': continue
    polarized = val['polarized']
    depolarized = val['depolarized']
    remained_polarized = val['remained_polarized']
    remained_nonpolarized = val['remained_nonpolarized']

    key_pieces = key.split(',')
    tactic = key_pieces[0]
    cit_dist = key_pieces[1]
    media_dist = key_pieces[2]

    latex_format_four_cats = latex_format_four_cats.replace(f'{tactic},{media_dist},{cit_dist}', f'{polarized}/{depolarized}/{remained_polarized}/{remained_nonpolarized}')
    latex_format_two_cats = latex_format_two_cats.replace(f'{tactic},{media_dist},{cit_dist}', f'{polarized+remained_polarized}/{depolarized+remained_nonpolarized}')

  with open(f'{data_dir}/{filename}'.replace('.tex','_all-categories.tex'),'w') as f:
    f.write(latex_format_four_cats)
  with open(f'{data_dir}/{filename}'.replace('.tex','_two-categories.tex'),'w') as f:
    f.write(latex_format_two_cats)
 
def polarization_stability_analysis(multidata, slope_threshold, intercept_threshold):
  '''
  Analyze each individual run of the polarization experiment
  to see if its individual runs polarization result match
  that of the mean of the results.

  This analysis supports the second to last paragraph of Section 4
  in Rabb & Cowen, 2022.

  :param multidata: Multidata gathered from the experiment.
  :param slope_threshold: A slope value to use to categorize results
  as polarizing or not based off of slope of a linear regression fit
  to their data.
  :param intercept_threshold: A y-value to use to categorize results
  as being polarized or not based off of the intercept of a linear
  regression fit to their data.
  '''
  # threshold = 0.0625
  # intercept = 6.25
  # slope_threshold = 0.01
  # intercept_threshold = 8.5
  stability_df = pd.DataFrame(columns=multidata['params'] + ['category','polarized','depolarized','remained_polarized','remained_nonpolarized','ratio_match'])

  polarization_data = { key: value for (key,value) in multidata.items() if key[1] == 'polarization' }
  polarization_means = { key: value['0'].mean(0) for (key,value) in polarization_data.items() }
  x = np.array([[val] for val in range(len(list(polarization_means.values())[0]))])

  # short_keys = list(polarization_data.keys())[0:10]
  # short_data = { key: val for (key,val) in polarization_data.items() if key in short_keys }
  for (param_combo, data) in polarization_data.items():
  # for (param_combo, data) in short_data:
    polarizing = []
    depolarizing = []
    remained_polarized = []
    remained_nonpolarized = []
    for run_data in data['0']:
      model = LinearRegression().fit(x, run_data)
      category = determine_polarization_categories(run_data, model, slope_threshold, intercept_threshold)
      if category == 'polarized':
        polarizing.append(run_data)
      elif category == 'depolarized':
        depolarizing.append(run_data)
      elif category == 'remained_polarized':
        remained_polarized.append(run_data)
      else:
        remained_nonpolarized.append(run_data)
    num_runs = len(data['0'])
    polarizing_ratio = len(polarizing) / num_runs
    depolarizing_ratio = len(depolarizing) / num_runs
    remain_polarized_ratio = len(remained_polarized) / num_runs
    remain_nonpolarized_ratio = len(remained_nonpolarized) / num_runs

    model = LinearRegression().fit(x, polarization_means[param_combo])
    category = determine_polarization_categories(polarization_means[param_combo], model, slope_threshold, intercept_threshold)

    category_to_ratio = {
      'polarized': polarizing_ratio,
      'depolarized': depolarizing_ratio,
      'remained_polarized': remain_polarized_ratio,
      'remained_nonpolarized': remain_nonpolarized_ratio
    }
    match = category_to_ratio[category]

    stability_df.loc[len(stability_df.index)] = list(param_combo[0]) + [category,polarizing_ratio,depolarizing_ratio,remain_polarized_ratio,remain_nonpolarized_ratio,match]
  
  diffs = stability_df['ratio_match'].unique()
  diff_parts = { diff: int((stability_df['ratio_match'] == diff).sum()) for diff in diffs }
  return { 'stability': stability_df, 'diff_parts': diff_parts }

def polarization_stability_across_repetitions(stability_df, multidata):
  '''
  Gather polarization stability data across different repetitions run in the
  BehaviorSpace simulation -- a given repetition is a different random graph
  generated with the same parameter set.

  :param stability_df: The dataframe of stability analysis broken down across
  repetitions.
  :param multidata: The original multidata from the experiment.
  '''
  polarization_data = { key: value for (key,value) in multidata.items() if key[1] == 'polarization' }
  params_minus_repetition = list_subtract(multidata['params'],['repetition'])
  stability_df_over_runs = pd.DataFrame(columns=params_minus_repetition + ['category','polarized','depolarized','remained_polarized','remained_nonpolarized','ratio_match'])
  polarization_data_across_repetitions = {}
  for key in polarization_data.keys():
    param_combo = key[0]
    combo_without_run = key[0][0:-1]
    if combo_without_run not in polarization_data_across_repetitions:
      query = ''
      for i in range(len(params_minus_repetition)):
        param = params_minus_repetition[i]
        val = param_combo[i]
        str_val = f'"{val}"'
        query += f"{param}=={val if type(val) != str else str_val} and "
      query = query[:-5]
      polarization_data_across_repetitions[combo_without_run] = stability_df.query(query)
      across_repetitions = polarization_data_across_repetitions[combo_without_run] 
      polarizing = across_repetitions['polarized'].sum() / len(across_repetitions)
      depolarizing = across_repetitions['depolarized'].sum() / len(across_repetitions)
      remained_polarized = across_repetitions['remained_polarized'].sum() / len(across_repetitions)
      remained_nonpolarized = across_repetitions['remained_nonpolarized'].sum() / len(across_repetitions)
      across_repetitions_dict = {'polarized': polarizing, 'depolarized': depolarizing, 'remained_polarized': remained_polarized, 'remained_nonpolarized': remained_nonpolarized}
      category = max(across_repetitions_dict, key=across_repetitions_dict.get)
      ratio_match = across_repetitions_dict[category]
      stability_df_over_runs.loc[len(stability_df_over_runs.index)] = list(combo_without_run) + [category,polarizing,depolarizing,remained_polarized,remained_nonpolarized,ratio_match]
  diffs = stability_df_over_runs['ratio_match'].unique()
  diff_parts = { diff: int((stability_df_over_runs['ratio_match'] == diff).sum()) for diff in diffs }
  return { 'stability': stability_df_over_runs, 'diff_parts': diff_parts }

def polarization_analysis_across_repetitions(polarization_df, multidata, slope_threshold, intercept_threshold):
  polarization_data = { key: value for (key,value) in multidata.items() if key[1] == 'polarization' }
  x = np.array([[val] for val in range(len(list(polarization_data.values())[0]['0'][0]))])
  params_minus_repetition = list_subtract(multidata['params'],['repetition'])
  df = pd.DataFrame(columns=params_minus_repetition + ['category','lr-intercept','lr-slope','var','start','end','delta','max','data'])

  polarization_data_across_repetitions = {}
  for key in polarization_data.keys():
    param_combo = key[0]
    combo_without_run = key[0][0:-1]
    if combo_without_run not in polarization_data_across_repetitions:
      query = ''
      for i in range(len(params_minus_repetition)):
        param = params_minus_repetition[i]
        val = param_combo[i]
        str_val = f'"{val}"'
        query += f"{param}=={val if type(val) != str else str_val} and "
      query = query[:-5]
      # query = f'translate=={param_combo[0]} and tactic=="{param_combo[1]}" and media_dist=="{param_combo[2]}" and citizen_dist=="{param_combo[4]}" and zeta_citizen=={param_combo[5]} and zeta_media=={param_combo[6]}'
      polarization_data_across_repetitions[combo_without_run] = polarization_df.query(query)
      across_runs = polarization_data_across_repetitions[combo_without_run]
      data = across_runs['data'].mean()
      model = LinearRegression().fit(x, data)
      category = determine_polarization_categories(data, model, slope_threshold, intercept_threshold)
      df.loc[len(df.index)] = list(combo_without_run) + [category,model.intercept_,model.coef_[0],data.var(),data[0],data[-1],data[-1]-data[0],max(data),data]

  polarizing = df[df['category'] == 'polarized']
  depolarizing = df[df['category'] == 'depolarized']
  remained_polarized = df[df['category'] == 'remained_polarized']
  remained_nonpolarized = df[df['category'] == 'remained_nonpolarized']

  return { 'polarization_df': df, 'polarized': polarizing, 'depolarized': depolarizing, 'remained_polarized': remained_polarized, 'remained_nonpolarized': remained_nonpolarized }

def determine_polarization_categories(data, model, slope_threshold, intercept_threshold):
  category = ''
  if model.coef_[0] >= slope_threshold:
    category = 'polarized'
  elif model.coef_[0] <= slope_threshold * -1:
    category = 'depolarized'
  else:
  # elif model.coef_[0] < slope_threshold and model.coef_[0] > slope_threshold * -1:
    if data[0] >= intercept_threshold and model.intercept_ < intercept_threshold: 
      category = 'depolarized'
    elif data[0] < intercept_threshold and model.intercept_ >= intercept_threshold:
      category = 'polarized'
    elif data[0] >= intercept_threshold and model.intercept_ >= intercept_threshold:
      category = 'remained_polarized'
    else:
      category = 'remained_nonpolarized'
  return category

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

def fragmentation_analysis(multidata):
  '''
  Gather data about fragmentation results from the multidata
  '''
  fragmentation_data = { key: value for (key,value) in multidata.items() if key[1] == 'fragmentation' }
  fragmentation_means = { key: value['default'].mean(0) for (key,value) in fragmentation_data.items() }
  fragmentation_vars = { key: value['default'].var(0).mean() for (key,value) in fragmentation_data.items() }
  mean_df = pd.DataFrame(columns=multidata['params'] + ['var','start','end','delta','max','data'])
  all_df = pd.DataFrame(columns=multidata['params'] + ['run','start','end','delta','max','data'])

  for (props, data) in fragmentation_means.items():
    mean_df.loc[len(mean_df.index)] = list(props[0]) + [fragmentation_vars[props],data[0],data[-1],data[-1]-data[0],max(data),data]

  for (props, data) in fragmentation_data.items():
    for i in range(len(data['default'])):
      run_data = data['default'][i]
      all_df.loc[len(all_df.index)] = list(props[0]) + [i,run_data[0],run_data[-1],run_data[-1]-run_data[0],max(run_data),run_data]
  
  return { 'fragmentation_all_df': all_df, 'fragmentation_mean_df': mean_df }

def homophily_analysis(multidata):
  homophily_data = { key: value for (key,value) in multidata.items() if key[1] == 'homophily' }
  homophily_means = { key: value['default'].mean(0) for (key,value) in homophily_data.items() }
  homophily_vars = { key: value['default'].var(0).mean() for (key,value) in homophily_data.items() }
  mean_df = pd.DataFrame(columns=multidata['params'] + ['var','start','end','delta','max','data'])
  all_df = pd.DataFrame(columns=multidata['params'] + ['run','start','end','delta','max','data'])

  for (props, data) in homophily_means.items():
    mean_df.loc[len(mean_df.index)] = list(props[0]) + [homophily_vars[props],data[0],data[-1],data[-1]-data[0],max(data),data]

  for (props, data) in homophily_data.items():
    for i in range(len(data['default'])):
      run_data = data['default'][i]
      all_df.loc[len(all_df.index)] = list(props[0]) + [i,run_data[0],run_data[-1],run_data[-1]-run_data[0],max(run_data),run_data]
  
  return { 'homophily_all_df': all_df, 'homophily_mean_df': mean_df }

def correlation_polarization_fragmentation_homophily_all(polarization_df, fragmentation_df, homophily_df, multidata):
  correlation_df = pd.DataFrame(columns=multidata['params'] + ['run','pf_correlation','ph_correlation'])
  runs_range = polarization_df['run'].unique()
  multidata_keys_minus_params = list_subtract(list(multidata.keys()), ['params'])
  param_combos = set(map(lambda key: key[0], multidata_keys_minus_params))
  for param_combo in param_combos:
    for run in runs_range:
      # print(f'Correlation for params {param_combo} run {run}')
      query = ''
      for i in range(len(multidata['params'])):
        param = multidata['params'][i]
        val = param_combo[i]
        str_val = f'"{val}"'
        query += f"{param}=={val if type(val) != str else str_val} and "
      query += f'run=={run}'
      # query = f'translate=={param_combo[0]} and tactic=="{param_combo[1]}" and media_dist=="{param_combo[2]}" and citizen_dist=="{param_combo[4]}" and zeta_citizen=={param_combo[5]} and zeta_media=={param_combo[6]} and citizen_memory_len=={param_combo[7]} and repetition=={param_combo[8]} and run=={run}'
      
      if len(polarization_df.query(query)) == 0:
        print(f'ERROR: No record in polarization for {param_combo} run {run}')
        continue

      polarization_row = polarization_df.query(query).iloc[0]
      fragmentation_row = fragmentation_df.query(query).iloc[0]
      homophily_row = homophily_df.query(query).iloc[0]
      pf_correlation_df = pd.DataFrame({'polarization': polarization_row['data'], 'fragmentation': fragmentation_row['data']})
      ph_correlation_df = pd.DataFrame({'polarization': polarization_row['data'], 'fragmentation': homophily_row['data']})
      pf_correlation = pf_correlation_df.corr(method='pearson')
      ph_correlation = ph_correlation_df.corr(method='pearson')
      correlation_df.loc[len(correlation_df.index)] = list(param_combo) + [run,pf_correlation.iloc[0,1],ph_correlation.iloc[0,1]]
  return correlation_df

def correlation_values_for_polarized(polarization_df, correlation_df):
  polarized = polarization_df[polarization_df['category']=='polarized']
  # 'category','lr-intercept','lr-slope','start','end','delta','max','data'
  merge_columns = list_subtract(polarization_df.columns, ['category','lr-intercept','lr-slope','start','end','delta','max','data'])
  p_df_with_corr = polarized.merge(correlation_df, on=merge_columns)
  print(f'pf correlation: mean {p_df_with_corr["pf_correlation"].mean()} var {p_df_with_corr["pf_correlation"].var()}')
  print(f'ph correlation: mean {p_df_with_corr["ph_correlation"].mean()} var {p_df_with_corr["ph_correlation"].var()}')
  return p_df_with_corr

def correlation_polarization_merge(polarization_df, correlation_df):
  # 'category','lr-intercept','lr-slope','start','end','delta','max','data'
  merge_columns = list_subtract(polarization_df.columns, ['category','lr-intercept','lr-slope','start','end','delta','max','data'])
  p_df_with_corr = polarization_df.merge(correlation_df, on=merge_columns)
  return p_df_with_corr


def plot_pol_frag_homo_correlation(correlation_df, out_path):
  polarization_categories = ['polarized','depolarized','remained_polarized','remained_nonpolarized']
  for category in polarization_categories:
    frag_data = correlation_df[correlation_df['category'] == category]['pf_correlation']
    homo_data = correlation_df[correlation_df['category'] == category]['ph_correlation']
    plt.figure()
    plt.scatter(frag_data,homo_data, s=1)
    plt.xlabel("Polarization x Fragmentation Correlation")
    plt.ylabel("Polarization x Homophily Correlation")
    plt.title("Per-Run Correlations of Polarization with Fragmentation & Homophily")
    plt.savefig(f'{out_path}/pol-frag-homo-correlation-scatter_{category}.png')
    plt.close()

def correlation_polarization_fragmentation_means(polarization_df, fragmentation_df, multidata, out_path):
  multidata_keys_minus_params = list(set(multidata.keys()) - set(['params']))
  param_combos = set(map(lambda key: key[0], multidata_keys_minus_params))
  data_to_plot = {
    'polarized': {'x': [], 'y': []},
    'depolarized': {'x': [], 'y': []},
    'remained_polarized': {'x': [], 'y': []},
    'remained_nonpolarized': {'x': [], 'y': []}
  }
  for param_combo in param_combos:
    query = ''
    for i in range(len(multidata['params'])):
      param = multidata['params'][i]
      val = param_combo[i]
      str_val = f'"{val}"'
      query += f"{param}=={val if type(val) != str else str_val} and "
    query = query[:-5]
    polarization_rows = polarization_df.query(query)
    fragmentation_rows = fragmentation_df.query(query)
    for i in polarization_rows.index:
      polarization_row = polarization_rows.loc[i]
      fragmentation_row = fragmentation_rows.loc[i]
      polarization_val = polarization_row['lr-slope']
      polarization_category = polarization_row['category']
      fragmentation_val = fragmentation_row['data'].mean()
      data_to_plot[polarization_category]['x'].append(fragmentation_val)
      data_to_plot[polarization_category]['y'].append(polarization_val)

  for (category,data) in data_to_plot.items():
    plt.figure()
    plt.scatter(data['x'],data['y'], label=category, s=1)
    x = np.array([ [x] for x in data['x'] ])
    model = LinearRegression().fit(x, data['y'])
    plt.plot(x, model.intercept_ + model.coef_ * x, color='red', label=rf'Regression $\alpha$={round(model.intercept_,3)}, $\beta$={round(model.coef_[0],3)}')
    plt.xlabel("Mean Fragmentation")
    plt.ylabel("Polarization")
    plt.legend()
    plt.savefig(out_path.replace('.png',f'-{category}.png'))
    plt.close()

  correlations = { category: pearsonr(data['x'],data['y']) for (category,data) in data_to_plot.items() }
  return correlations

def correlation_polarization_homophily_means(polarization_df, homophily_df, multidata, out_path):
  multidata_keys_minus_params = list(set(multidata.keys()) - set(['params']))
  param_combos = set(map(lambda key: key[0], multidata_keys_minus_params))
  data_to_plot = {
    'polarized': {'x': [], 'y': []},
    'depolarized': {'x': [], 'y': []},
    'remained_polarized': {'x': [], 'y': []},
    'remained_nonpolarized': {'x': [], 'y': []}
  }
  for param_combo in param_combos:
    query = ''
    for i in range(len(multidata['params'])):
      param = multidata['params'][i]
      val = param_combo[i]
      str_val = f'"{val}"'
      query += f"{param}=={val if type(val) != str else str_val} and "
    query = query[:-5]
    # query = f'translate=={param_combo[0]} and tactic=="{param_combo[1]}" and media_dist=="{param_combo[2]}" and citizen_dist=="{param_combo[4]}" and zeta_citizen=={param_combo[5]} and zeta_media=={param_combo[6]} and repetition=={param_combo[8]}'
    polarization_rows = polarization_df.query(query)
    homophily_rows = homophily_df.query(query)
    for i in polarization_rows.index:
      polarization_row = polarization_rows.loc[i]
      homophily_row = homophily_rows.loc[i]
      polarization_val = polarization_row['lr-slope']
      polarization_category = polarization_row['category']
      homophily_val = homophily_row['data'].mean()
      data_to_plot[polarization_category]['x'].append(homophily_val)
      data_to_plot[polarization_category]['y'].append(polarization_val)

  for (category,data) in data_to_plot.items():
    plt.figure()
    plt.scatter(data['x'],data['y'], label=category, s=1)
    x = np.array([ [x] for x in data['x'] ])
    model = LinearRegression().fit(x, data['y'])
    plt.plot(x, model.intercept_ + model.coef_ * x, color='red', label=rf'Regression $\alpha$={round(model.intercept_,3)}, $\beta$={round(model.coef_[0],3)}')
    plt.xlabel("Mean Homophily")
    plt.ylabel("Polarization")
    plt.legend()
    plt.savefig(out_path.replace('.png',f'-{category}.png'))
    plt.close()

  correlations = { category: pearsonr(data['x'],data['y']) for (category,data) in data_to_plot.items() }
  return correlations

def polarizing_results_analysis(df):
  '''
  One specific analysis that supports Table 3 in the Rabb & Cowen
  paper on a static ecosystem cascade model. It returns the proportion
  of results within result partitions by institution tactic, when
  parameters epsilon, gamma, and h_G are set certain ways.
  '''
  tactics = ['broadcast-brain', 'appeal-mean', 'appeal-median', 'appeal-mode']
  for tactic in tactics:
    total = len(df[df['tactic'] == tactic])
    print(f'{tactic} ({total})')
    print(len(df.query(f'tactic == "{tactic}" and epsilon=="0"')) / total)
    print(len(df.query(f'tactic == "{tactic}" and epsilon=="1"')) / total)
    print(len(df.query(f'tactic == "{tactic}" and epsilon=="2"')) / total)
    print(len(df.query(f' tactic == "{tactic}" and translate=="0"')) / total)
    print(len(df.query(f'tactic == "{tactic}" and translate=="1"')) / total)
    print(len(df.query(f'tactic == "{tactic}" and translate=="2"')) / total)
    print(len(df.query(f'tactic == "{tactic}" and graph_type=="ba-homophilic"')) / total)
    print(len(df.query(f'tactic == "{tactic}" and graph_type=="barabasi-albert"')) / total)

def polarizing_results_analysis_by_param(df, params):
  '''
  Query polarization results by all parameter combinations
  and return results.

  :param df: The experiment results data frame for polarization
  results.
  :param params: A dictionary of parameters to use for
  segmentation during results analysis (currently hardcoded below).
  '''
  params = {
    'translate' : ['0', '1', '2'],
    'epsilon' : ['0', '1', '2'],
    'tactic' : ['broadcast-brain', 'appeal-mean', 'appeal-median', 'appeal-mode'],
    'media_dist' : [ 'uniform', 'normal', 'polarized' ],
    'graph_type' : [ 'ba-homophilic', 'barabasi-albert' ],
    'citizen_dist' : ['normal', 'uniform', 'polarized'],
  }

  num_params = 6
  all_params = []
  for param in params.keys():
    param_list = params[param]
    for val in param_list:
      all_params.append((param, val))

  combos = {}
  for combo_len in range(1, num_params):
    for combo in itertools.combinations(all_params, combo_len):
      unique_keys = set([pair[0] for pair in combo])
      flat_combo = ({key: [pair[1] for pair in combo if pair[0] == key] for key in unique_keys})
      combos[len(combos)] = flat_combo
    # for combo in itertools.product(*param_combos, combo_len):
    #   combos.append(combo)
  
  ratios = {}
  param_dfs = {}
  for combo_i in combos.keys():
    combo = combos[combo_i]
    query = '('
    for param in combo.keys():
      query += f'('
      for val in combo[param]:
        query += f'{param}=="{val}" or '
      query = query[:-4] + ') and '
    query = query[:-5] + ')'
    param_dfs[combo_i] = df.query(query)
    ratios[combo_i] = len(param_dfs[combo_i]) / len(df)
  
  return (combos, param_dfs, ratios)

  # return combos