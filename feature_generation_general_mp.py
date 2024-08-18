import logging
import pandas as pd
import time
from multiprocessing import Pool

def add_flow_number_column(pcap_df):
    pcap_df['Flow'] = pcap_df['Source'] + '->' + pcap_df['Destination']
    pcap_df['FlowNumber'] = 0
    flows = {flow_key: 0 for flow_key in pcap_df['Flow'].unique()}
    for index, row in pcap_df.iterrows():
        pcap_df.at[index, 'FlowNumber'] = flows[row['Flow']]
        flows[row['Flow']] += 1
    return pcap_df

def aggregate_sequences(df, seq_length, window_size, mean):
    aggregated_data = []

    df_range = []
    if mean == "packets":
        df_range = range(0, len(df) - seq_length + 1, window_size)
    elif mean == "time":
        df_range = range(0, 60*30-seq_length, window_size)
    
    for start in df_range:
        if mean == "packets":
            sequence = df.iloc[start:start + seq_length]
            seq_seconds = sequence['Time'].iloc[seq_length-1] - sequence['Time'].iloc[0]
            seq_packets = seq_length
        elif mean == "time":
            sequence = df[df['Time'] >= start]
            sequence = sequence[sequence['Time'] < start + seq_length]
            seq_seconds = seq_length
            seq_packets = len(sequence)
            if seq_packets == 0:
                continue
        
        length_sum = sequence['Length'].sum()
        byte_rate = length_sum/seq_seconds
        packet_rate = seq_packets/seq_seconds
        avg_flow_number = sequence['FlowNumber'].mean()
        avg_deltatime = sequence['Delta-Time'].mean()
        num_modbus = (sequence['Protocol'] == 'Modbus/TCP').sum()
        num_tcp = (sequence['Protocol'] == 'TCP').sum()
        num_ping = (sequence['Protocol'] == 'ICMP').sum()
        num_other = seq_packets - num_modbus - num_tcp - num_ping
        num_syn = (sequence['SYN'] == 'Set').sum()
        num_ack = (sequence['ACK'] == 'Set').sum()
        modbus_rate = num_modbus/seq_length
        tcp_rate = num_tcp/seq_packets
        ping_rate = num_ping/seq_packets
        other_rate = num_other/seq_packets
        if num_syn == 0:
            synack_ratio = 0
        elif num_ack == 0:
            synack_ratio = 1
        else:
            synack_ratio = num_syn/num_ack
        source_entropy = sequence['Source'].nunique()/seq_packets
        destination_entropy = sequence['Destination'].nunique()/seq_packets
        s_port_entropy = sequence['S-Port'].nunique()/seq_packets
        d_port_entropy = sequence['D-Port'].nunique()/seq_packets
        label = sequence['Label'].iloc[0]
        
        aggregated_row = {
                'byte_rate' : byte_rate,
                'packet_rate' : packet_rate,
                'avg_flow_number' : avg_flow_number,
                'avg_deltatime' : avg_deltatime,
                'source_entropy' : source_entropy,
                'destination_entropy' : destination_entropy,
                's-port_entropy': s_port_entropy,
                'd-port_entropy': d_port_entropy,
                'synack_ratio' : synack_ratio,
                'modbus_rate' : modbus_rate,
                'tcp_rate' : tcp_rate,
                'ping_rate' : ping_rate,
                'other_rate' : other_rate,
                'label' : label
        }
        
        aggregated_data.append(aggregated_row)

    aggregated_df = pd.DataFrame(aggregated_data)
    
    return aggregated_df

def aggregate_df(seq_length, mean):
    window_size = seq_length
    aggregated_dfs = []
    for df in dfs:
        agg_df = aggregate_sequences(df, seq_length, window_size, mean)
        aggregated_dfs.append(agg_df)

    aggregated_df = pd.concat(aggregated_dfs)
    if mean == "packets":
        print(f'CREATED csv/aggregated_{seq_length}packets_df.csv')
        aggregated_df.to_csv(f'csv/aggregated_{seq_length}packets_df.csv', index=False)
    elif mean == "time":
        print(f'CREATED csv/aggregated_time_{seq_length}s_df.csv')
        aggregated_df.to_csv(f'csv/aggregated_time_{seq_length}s_df.csv', index=False)

clean_df = pd.read_csv('csv/clean15-30.csv')
tcpsynflood_df = pd.read_csv('csv/tcpsynflood15-30.csv')
pingflood_df = pd.read_csv('csv/pingflood15-30.csv')
modbusflood_df = pd.read_csv('csv/modbusflood15-30.csv')

clean_df['Label'] = 0
tcpsynflood_df['Label'] = 1
pingflood_df['Label'] = 2
modbusflood_df['Label'] = 3

dfs = [clean_df, tcpsynflood_df, pingflood_df, modbusflood_df]

packet_seq_lengths = [10,20,50,100,250,500]
time_seq_lengths = [1,2,5,10,30]
threads = []

print("Generating flow numbers")

start = time.time()
pool1 = Pool(processes=len(dfs))

df_results = []
for df in dfs:
    df_results.append(pool1.apply_async(add_flow_number_column, [df]))

pool1.close()
pool1.join()

for i in range(len(dfs)):
    dfs[i] = df_results[i].get()

print("Generating and starting processes")
pool2 = Pool(processes=(len(packet_seq_lengths)+len(time_seq_lengths)))
packet_res = []
for seq_length in packet_seq_lengths:
    packet_res.append(pool2.apply_async(aggregate_df, [seq_length,"packets"]))
print(f"Started processes for packet sequences with length {packet_seq_lengths}")
time_res = []
for seq_length in time_seq_lengths:
    time_res.append(pool2.apply_async(aggregate_df, [seq_length,"time"]))
print(f"Started processes for time sequences with length {time_seq_lengths}")

pool2.close()
print("Waiting until all processes are done")
pool2.join()
end = time.time()

for res in (packet_res + time_res):
    if res.get() != None:
        print(res.get())

print("CSVs created!")
print(f"Execution time: {end-start} seconds")
