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

    for start in range(0, 60*30-seq_length, window_size):
        
        sequence = df[df['Time'] >= start]
        sequence = sequence[sequence['Time'] < start + seq_length]
        
        if (len(sequence)==0):
            aggregated_row = {
                    'byte_rate' : 0,
                    'packet_rate' : 0,
                    'avg_flow_number' : 0,
                    'avg_deltatime' : seq_length,
                    'source_entropy' : 1,
                    'destination_entropy' : 1,
                    's-port_entropy': 1,
                    'd_port_entropy': 1,
                    'synack_ratio' : 0,
                    'modbus_rate' : 0,
                    'tcp_rate' : 0,
                    'ping_rate' : 0,
                    'other_rate' : 0,
                    'label' : label
            }
        
        else:
            length_sum = sequence['Length'].sum()
            packet_number = len(sequence)
            byte_rate = length_sum/seq_length
            packet_rate = len(sequence)/seq_length
            avg_flow_number = sequence['FlowNumber'].mean()
            avg_deltatime = sequence['Delta-Time'].mean()
            num_modbus = (sequence['Protocol'] == 'Modbus/TCP').sum()
            num_tcp = (sequence['Protocol'] == 'TCP').sum()
            num_ping = (sequence['Protocol'] == 'ICMP').sum()
            num_other = packet_number - num_modbus - num_tcp - num_ping
            num_syn = (sequence['SYN'] == 'Set').sum()
            num_ack = (sequence['ACK'] == 'Set').sum()
            modbus_rate = num_modbus/packet_number
            tcp_rate = num_tcp/packet_number
            ping_rate = num_ping/packet_number
            other_rate = num_other/packet_number
            if num_syn == 0:
                synack_ratio = 0
            elif num_ack == 0:
                synack_ratio = 1
            else:
                synack_ratio = num_syn/num_ack
            source_entropy = sequence['Source'].nunique()/packet_number
            destination_entropy = sequence['Destination'].nunique()/packet_number
            s_port_entropy = sequence['S-Port'].nunique()/packet_number
            d_port_entropy = sequence['D-Port'].nunique()/packet_number
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

for seq_length in [1, 2, 5, 10, 30]:
    window_size = seq_length

    clean_agg_df = aggregate_sequences(clean_df, seq_length, window_size)
    tcpsynflood_agg_df = aggregate_sequences(tcpsynflood_df, seq_length, window_size)
    pingflood_agg_df = aggregate_sequences(pingflood_df, seq_length, window_size)
    modbusflood_agg_df = aggregate_sequences(modbusflood_df, seq_length, window_size)
    aggregated_df = pd.concat([clean_agg_df, tcpsynflood_agg_df, pingflood_agg_df, modbusflood_agg_df])

    print(aggregated_df)


    aggregated_df.to_csv(f'csv/aggregated_time_{seq_length}s_df.csv', index=False)

