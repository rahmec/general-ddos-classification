import pandas as pd

clean_df = pd.read_csv('csv/clean15-30.csv')
tcpsynflood_df = pd.read_csv('csv/tcpsynflood15-30.csv')
pingflood_df = pd.read_csv('csv/pingflood15-30.csv')
modbusflood_df = pd.read_csv('csv/modbusflood15-30.csv')

clean_df['Label'] = 0
tcpsynflood_df['Label'] = 1
pingflood_df['Label'] = 2
modbusflood_df['Label'] = 3

flows_count = []

def add_flow_number_column(pcap_df):
  pcap_df['Flow'] = pcap_df['Source'] + '->' + pcap_df['Destination']
  pcap_df['FlowNumber'] = 0
  flows = {flow_key: 0 for flow_key in pcap_df['Flow'].unique()}
  for index, row in pcap_df.iterrows():
    pcap_df.at[index, 'FlowNumber'] = flows[row['Flow']]
    flows[row['Flow']] += 1

add_flow_number_column(clean_df)
add_flow_number_column(tcpsynflood_df)
add_flow_number_column(pingflood_df)
add_flow_number_column(modbusflood_df)

def aggregate_sequences(df, seq_length, window_size):
    aggregated_data = []

    for start in range(0, len(df) - seq_length + 1, window_size):
        sequence = df.iloc[start:start + seq_length]
        
        length_sum = sequence['Length'].sum()
        time_length = sequence['Time'].iloc[seq_length-1] - sequence['Time'].iloc[0]
        byte_rate = length_sum/time_length
        packet_rate = seq_length/time_length
        avg_flow_number = sequence['FlowNumber'].mean()
        avg_deltatime = sequence['Delta-Time'].mean()
        num_modbus = (sequence['Protocol'] == 'Modbus/TCP').sum()
        num_tcp = (sequence['Protocol'] == 'TCP').sum()
        num_ping = (sequence['Protocol'] == 'ICMP').sum()
        num_other = seq_length - num_modbus - num_tcp - num_ping
        num_syn = (sequence['SYN'] == 'Set').sum()
        num_ack = (sequence['ACK'] == 'Set').sum()
        modbus_rate = num_modbus/seq_length
        tcp_rate = num_tcp/seq_length
        ping_rate = num_ping/seq_length
        other_rate = num_other/seq_length
        if num_syn == 0:
            synack_ratio = 0
        elif num_ack == 0:
            synack_ratio = 1
        else:
            synack_ratio = num_syn/num_ack
        source_entropy = sequence['Source'].nunique()/seq_length
        destination_entropy = sequence['Destination'].nunique()/seq_length
        s_port_entropy = sequence['S-Port'].nunique()/seq_length
        d_port_entropy = sequence['D-Port'].nunique()/seq_length
        label = sequence['Label'].iloc[0]
        
        aggregated_row = {
                'byte_rate' : byte_rate,
                'packet_rate' : packet_rate,
                'avg_flow_number' : avg_flow_number,
                'avg_deltatime' : avg_deltatime,
                'source_entropy' : source_entropy,
                'destination_entropy' : destination_entropy,
                's-port_entropy': s_port_entropy,
                'd_port_entropy': d_port_entropy,
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

#lens = [10,20,50,100,250]
lens = [500]

for seq_length in lens:
    window_size = seq_length

    clean_agg_df = aggregate_sequences(clean_df, seq_length, window_size)
    tcpsynflood_agg_df = aggregate_sequences(tcpsynflood_df, seq_length, window_size)
    pingflood_agg_df = aggregate_sequences(pingflood_df, seq_length, window_size)
    modbusflood_agg_df = aggregate_sequences(modbusflood_df, seq_length, window_size)
    aggregated_df = pd.concat([clean_agg_df, tcpsynflood_agg_df, pingflood_agg_df, modbusflood_agg_df])

    aggregated_df.to_csv(f'csv/aggregated_{seq_length}packets_df.csv', index=False)
