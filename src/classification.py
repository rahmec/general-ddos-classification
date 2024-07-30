from re import S
import numpy as np
import pandas as pd
#import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load your datasets
clean_df = pd.read_csv('clean15-30.csv')
tcpsynflood_df = pd.read_csv('tcpsynflood15-30.csv')
pingflood_df = pd.read_csv('pingflood15-30.csv')
modbusflood_df = pd.read_csv('modbusflood15-30.csv')

# Assign labels to each dataset
clean_df['Label'] = 0
tcpsynflood_df['Label'] = 1
pingflood_df['Label'] = 2
modbusflood_df['Label'] = 3

# Concatenate all datasets into one DataFrame
#df = pd.concat([clean_df, tcpflood_df, pingflood_df])
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

# Function to aggregate sequences of packets with a sliding window
def aggregate_sequences(df, seq_length, window_size):
    # Create an empty list to store aggregated data
    aggregated_data = []

    for start in range(0, len(df) - seq_length + 1, window_size):
        # Extract the window
        sequence = df.iloc[start:start + seq_length]
        
        # Calculate aggregated features
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
        source_entropy = sequence['Source'].value_counts()/seq_length
        destination_entropy = sequence['Destination'].value_counts()/seq_length
        s_port_entropy = sequence['S-Port'].value_counts()/seq_length
        d_port_entropy = sequence['D-Port'].value_counts()/seq_length
        label = sequence['Label'].iloc[0]
        # Create a dictionary of aggregated features
        aggregated_row = {
                'time_length' : time_length,
                'byte_rate' : byte_rate,
                'packet_rate' : packet_rate, 'avg_flow_number' : avg_flow_number,
                'avg_deltatime' : avg_deltatime,
                'source_entropy' : source_entropy,
                'destination_entropy' : destination_entropy,
                'sport_entropy': s_port_entropy,
                'd_port_entropy': d_port_entropy,
                'synack_ratio' : synack_ratio,
                'modbus_rate' : modbus_rate,
                'tcp_rate' : tcp_rate,
                'ping_rate' : ping_rate,
                'other_rate' : other_rate,
                'label' : label
        }
        
        # Append the dictionary to the list
        aggregated_data.append(aggregated_row)

    # Convert the list of dictionaries to a DataFrame
    aggregated_df = pd.DataFrame(aggregated_data)
    
    return aggregated_df

# Specify the sequence length (stride) and window size
seq_length = 20
window_size = 20

# Create the aggregated DataFrame
clean_agg_df = aggregate_sequences(clean_df, seq_length, window_size)
tcpsynflood_agg_df = aggregate_sequences(tcpsynflood_df, seq_length, window_size)
pingflood_agg_df = aggregate_sequences(pingflood_df, seq_length, window_size)
modbusflood_agg_df = aggregate_sequences(modbusflood_df, seq_length, window_size)
aggregated_df = pd.concat([clean_agg_df, tcpsynflood_agg_df, pingflood_agg_df, modbusflood_agg_df])

aggregated_df.to_csv('aggregated_df.csv', index=False)


# ModbusString = 'Modbus/TCP'
# TCPString = 'TCP'
# PingString = 'ICMP'


'''
for row in df:
  flow = df['Source'] + '->' + df['Destination']

df['Flow'] = df['Source'] + '->' + df['Destination']
# Calculate the number of packets per flow
flow_packet_counts = df.groupby('Flow').size().reset_index(name='Packet Count per Flow')
print(flow_packet_counts)

# Encode into numerical values
encoder = LabelEncoder()
df['Protocol'] = encoder.fit_transform(df['Protocol'])
df['Source'] = encoder.fit_transform(df['Source'])
df['TCPFlag'] = encoder.fit_transform(df['TCPFlag'])
df['SeqNum'] = encoder.fit_transform(df['SeqNum'])
df['Destination'] = encoder.fit_transform(df['Destination'])


# Calculate packet rate

# Define sequence length
seq_length = 30

# Function to create sequences
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

# Create sequences for each feature
time_sequences = create_sequences(df['Delta-Time'].values, seq_length)
length_sequences = create_sequences(df['Length'].values, seq_length)
protocol_sequences = create_sequences(df['Protocol'].values, seq_length)
sport_sequences = create_sequences(df['S-Port'].values, seq_length)
dport_sequences = create_sequences(df['D-Port'].values, seq_length)
sip_sequences = create_sequences(df['Source'].values, seq_length)
dip_sequences = create_sequences(df['Destination'].values, seq_length)
seq_sequences = create_sequences(df['SeqNum'].values, seq_length)
flag_sequences = create_sequences(df['TCPFlag'].values, seq_length)

#print("Shape of time_sequences:", time_sequences.shape)  # Should be (num_sequences, seq_length)
#print("Shape of length_sequences:", length_sequences.shape)  # Should be (num_sequences, seq_length)
#print("Shape of protocol_sequences:", protocol_sequences.shape)  # Should be (num_sequences, seq_length
# Combine the sequences into a single tensor with shape (num_sequences, num_channels, seq_length)
sequences_tensor = torch.tensor(np.stack((time_sequences, length_sequences, protocol_sequences, sport_sequences, dport_sequences, sip_sequences,dip_sequences,seq_sequences,flag_sequences), axis=1), dtype=torch.float32)
#sequences_tensor = torch.tensor(np.stack((time_sequences, length_sequences, protocol_sequences), axis=1), dtype=torch.float32)

#print("Shape of sequences_tensor:", sequences_tensor.shape)

# Create sequences for labels
labels = df['Label'].values
labels_seq = create_sequences(labels, seq_length)
labels_tensor = torch.tensor(labels_seq[:, -1], dtype=torch.long)  # Use the last element of each sequence as the label

#print("Shape of labels_tensor:", labels_tensor.shape)

# Split data into training and testing sets
#X_train, X_test, Y_train, Y_test = train_test_split(sequences_tensor, labels_tensor, test_size=0.2, shuffle=True, stratify=labels_tensor, random_state=42)

#print("Shape of X_train:", X_train.shape)
#print("Shape of Y_train:", Y_train.shape)
#print("Shape of X_test:", X_test.shape)
#print("Shape of Y_test:", Y_test.shape)


# Initialize K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

for train_index, test_index in kf.split(sequences_tensor):
    X_train, X_test = sequences_tensor[train_index], sequences_tensor[test_index]
    Y_train, Y_test = labels_tensor[train_index], labels_tensor[test_index]

    X_train_flat = X_train.view(X_train.shape[0], -1)
    X_test_flat = X_test.view(X_test.shape[0], -1)

    nan_mask = np.isnan(X_train_flat.numpy())
    X_train_flat[nan_mask] = -1

    nan_mask_test = np.isnan(X_test_flat.numpy())
    X_test_flat[nan_mask_test] = -1

    clf =  RandomForestClassifier()
    clf.fit(X_train_flat, Y_train)

    # Evaluate the classifier
    Y_pred = clf.predict(X_test_flat)
    acc = accuracy_score(Y_test, Y_pred)
    accuracies.append(acc)

    print(f"Fold Accuracy: {acc}")

print(f"Mean Accuracy: {np.mean(accuracies)}")
print(f"Standard Deviation of Accuracy: {np.std(accuracies)}")

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Carica i dataset
clean_df = pd.read_csv('/content/clean-0,5h.csv')
tcpflood_df = pd.read_csv('/content/tcpflood_15m_0,5h.csv')
pingflood_df = pd.read_csv('/content/pingflood_15m_0,5h.csv')

# Assegna etichette ai dataset
clean_df['Label'] = 0
tcpflood_df['Label'] = 1
pingflood_df['Label'] = 2

# Concatenare tutti i dataset in un DataFrame
df = pd.concat([clean_df, tcpflood_df, pingflood_df])

# Codifica delle caratteristiche in valori numerici
encoder = LabelEncoder()
df['Protocol'] = encoder.fit_transform(df['Protocol'])
df['Source'] = encoder.fit_transform(df['Source'])
df['TCPFlag'] = encoder.fit_transform(df['TCPFlag'])
df['SeqNum'] = encoder.fit_transform(df['SeqNum'])
df['Destination'] = encoder.fit_transform(df['Destination'])

# Rimuovi le caratteristiche S-Port e D-Port
df = df.drop(columns=['S-Port', 'D-Port'])

# Definisci la funzione per creare sequenze
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

# Funzione per elaborare i dati per una data lunghezza di sequenza
def process_data_for_seq_length(df, seq_length):
    # Crea sequenze per ogni caratteristica
    time_sequences = create_sequences(df['Delta-Time'].values, seq_length)
    length_sequences = create_sequences(df['Length'].values, seq_length)
    protocol_sequences = create_sequences(df['Protocol'].values, seq_length)
    sip_sequences = create_sequences(df['Source'].values, seq_length)
    dip_sequences = create_sequences(df['Destination'].values, seq_length)
    seq_sequences = create_sequences(df['SeqNum'].values, seq_length)
    flag_sequences = create_sequences(df['TCPFlag'].values, seq_length)

    # Combina le sequenze in un singolo tensor con forma (num_sequences, num_channels, seq_length)
    sequences_tensor = torch.tensor(np.stack((time_sequences, length_sequences, protocol_sequences, sip_sequences, dip_sequences, seq_sequences, flag_sequences), axis=1), dtype=torch.float32)

    # Crea sequenze per le etichette
    labels = df['Label'].values
    labels_seq = create_sequences(labels, seq_length)
    labels_tensor = torch.tensor(labels_seq[:, -1], dtype=torch.long)  # Usa l'ultimo elemento di ogni sequenza come etichetta

    return sequences_tensor, labels_tensor

# Prova diverse lunghezze di sequenza
seq_lengths = [7, 15, 30, 40, 60]

for seq_length in seq_lengths:
    print(f"Processing data for seq_length = {seq_length}")
    sequences_tensor, labels_tensor = process_data_for_seq_length(df, seq_length)

    # Inizializza Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []

    for train_index, test_index in skf.split(sequences_tensor, labels_tensor):
        X_train, X_test = sequences_tensor[train_index], sequences_tensor[test_index]
        Y_train, Y_test = labels_tensor[train_index], labels_tensor[test_index]

        # Flatten the data for the classifier
        X_train_flat = X_train.view(X_train.shape[0], -1)
        X_test_flat = X_test.view(X_test.shape[0], -1)

        # Train
        classifier = DecisionTreeClassifier(random_state=42)
        classifier.fit(X_train_flat.numpy(), Y_train.numpy())

        # Evaluate the classifier
        Y_pred = classifier.predict(X_test_flat.numpy())
        acc = accuracy_score(Y_test.numpy(), Y_pred)
        accuracies.append(acc)

        print(f"Fold {train_index} Accuracy: {acc}")

        # Calcola la matrice di confusione
        conf_matrix = confusion_matrix(Y_test.numpy(), Y_pred)

        # Visualizza la matrice di confusione
        plt.figure(figsize=(7, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Clean', 'TCP Flood', 'Ping Flood'], yticklabels=['Clean', 'TCP Flood', 'Ping Flood'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

    print(f"Mean Accuracy for seq_length = {seq_length}: {np.mean(accuracies)}")
    print(f"Standard Deviation of Accuracy for seq_length = {seq_length}: {np.std(accuracies)}")
    print("\n---------------------------------------------------------------\n")

# Seleziona le caratteristiche di interesse
features = ['Delta-Time', 'Length', 'Protocol', 'S-Port', 'D-Port','Source','Destination','SeqNum','TCPFlag']
df = df[features]

# Calcola la matrice di correlazione usando il metodo di Pearson
correlation_matrix = df.corr(method='pearson')

import seaborn as sns
import matplotlib.pyplot as plt

# Visualizza la matrice di correlazione usando una heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Matrice di Correlazione')
plt.show()

sequences_tensor[0][2][:]

X_train[0][2][:]

# Contare i NaN nei dati originali
X_train_np = X_train.numpy()
nan_count_original = np.isnan(X_train_np).sum()
print(f"Numero di NaN nei dati originali: {nan_count_original}")

# Flatten = appiattire the sequences for scaling and classifier compatibility
X_train_flat = X_train.view(X_train.shape[0], -1)
X_test_flat = X_test.view(X_test.shape[0], -1)

# Convert the flattened data back to PyTorch tensors
X_train_flat_np = X_train_flat.numpy()
X_test_flat_np = X_test_flat.numpy()

# Apply SparseScaler
#scaler = SparseScaler()
#X_train_scaled = scaler.fit_transform(X_train_flat)
#X_test_scaled = scaler.transform(X_test_flat)

# Convert the scaled data back to numpy arrays for classifier compatibility
#X_train_scaled_np = X_train_scaled.numpy()
#X_test_scaled_np = X_test_scaled.numpy()

# Verifica il numero di NaN dopo il flattening
nan_mask_flattened = np.isnan(X_train_flat.numpy())
print(f"Numero di NaN dopo il flattening: {nan_mask_flattened.sum()}")

# metodo 1
# Check for NaN values and handle them (e.g., replace NaNs with the mean of the column)

nan_mask = np.isnan(X_train_flat_np)
#X_train_flat_np[nan_mask] = np.nanmean(X_train_flat_np)
X_train_flat_np[nan_mask] = -1

nan_mask_test = np.isnan(X_test_flat_np)
#X_test_flat_np[nan_mask_test] = np.nanmean(X_test_flat_np)
X_test_flat_np[nan_mask_test] = -1

nan_finale = np.isnan(X_train_flat_np)
print(f"Numero di NaN dopo il replace: {nan_finale.sum()}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Initialize and train the classifier
classifier = RandomForestClassifier()
#classifier = KNeighborsClassifier(n_neighbors=50)
#classifier = DecisionTreeClassifier(random_state=42)
#classifier = SVC(kernel='linear', random_state=42)
#classifier = LinearSVC(random_state=42)

# Training using the hydra.
#classifier.fit(X_train_transform, Y_train)

# Training using the scaled data without hydra and NaNs.
classifier.fit(X_train_flat_np, Y_train.numpy())

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Make predictions on the test set
#predictions = classifier.predict(X_test_transform)
predictions = classifier.predict(X_test_flat_np)

# Evaluate the classifier
accuracy = accuracy_score(Y_test.numpy(), predictions)
print("Test Accuracy:", accuracy)
print(classification_report(Y_test.numpy(), predictions, target_names=["Clean", "TCP Flood", "Ping Flood"]))

# Calcola la matrice di confusione
conf_matrix = confusion_matrix(Y_test.numpy(), predictions)

# Visualizza la matrice di confusione
plt.figure(figsize=(7, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Clean', 'TCP Flood', 'Ping Flood'], yticklabels=['Clean', 'TCP Flood', 'Ping Flood'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Get feature importances
importances = classifier.feature_importances_
feature_importances = pd.DataFrame({'Feature': X_train_flat_np.columns, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance using Random Forest')
plt.gca().invert_yaxis()  # To display the most important features at the top
plt.show()

print("Random Forest Feature Importances:")
print(feature_importances)

from sklearn.model_selection import cross_val_score

# Lista di possibili lunghezze di sequenza
seq_lengths = [9, 15, 20, 30]

# Dizionario per salvare le performance
performance = {}

for seq_length in seq_lengths:
    # Crea le sequenze per ciascuna lunghezza
    time_sequences = create_sequences(df['Delta-Time'].values, seq_length)
    length_sequences = create_sequences(df['Length'].values, seq_length)
    protocol_sequences = create_sequences(df['Protocol'].values, seq_length)

    sequences_tensor = torch.tensor(np.stack((time_sequences, length_sequences, protocol_sequences), axis=1), dtype=torch.float32)

    labels_seq = create_sequences(labels, seq_length)
    labels_tensor = torch.tensor(labels_seq[:, -1], dtype=torch.long)

    X_train, X_test, Y_train, Y_test = train_test_split(sequences_tensor, labels_tensor, test_size=0.2, shuffle=True, random_state=42)

    X_train_flat = X_train.view(X_train.shape[0], -1).numpy()
    X_test_flat = X_test.view(X_test.shape[0], -1).numpy()

    # Gestione dei valori NaN
    X_train_flat[np.isnan(X_train_flat)] = -1
    X_test_flat[np.isnan(X_test_flat)] = -1

    # Standardizzazione dei dati
    #scaler = StandardScaler()
    #X_train_scaled = scaler.fit_transform(X_train_flat)
    #X_test_scaled = scaler.transform(X_test_flat)

    # Modello di classificazione
    classifier = DecisionTreeClassifier(random_state=42)

    # Valutazione del modello con cross-validation
    scores = cross_val_score(classifier, X_train_flat, Y_train.numpy(), cv=5)
    performance[seq_length] = scores.mean()

# Seleziona la lunghezza di sequenza con la migliore performance
best_seq_length = max(performance, key=performance.get)
print(f"Best sequence length: {best_seq_length}")

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load your datasets
clean_df = pd.read_csv('/content/clean-0,5h.csv')
tcpflood_df = pd.read_csv('/content/tcpflood_15m_0,5h.csv')
pingflood_df = pd.read_csv('/content/pingflood_15m_0,5h.csv')

# Assign labels to each dataset
clean_df['Label'] = 0
tcpflood_df['Label'] = 1
pingflood_df['Label'] = 2

# Concatenate all datasets into one DataFrame
df = pd.concat([clean_df, tcpflood_df, pingflood_df])

# Encode 'Protocol' into numerical values
protocol_encoder = LabelEncoder()
df['Protocol'] = protocol_encoder.fit_transform(df['Protocol'])

# Define function to create sequences
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

# Function to process data for a given seq_length
def process_data_for_seq_length(df, seq_length):
    # Create sequences for each feature
    time_sequences = create_sequences(df['Delta-Time'].values, seq_length)
    length_sequences = create_sequences(df['Length'].values, seq_length)
    protocol_sequences = create_sequences(df['Protocol'].values, seq_length)

    # Combine the sequences into a single tensor with shape (num_sequences, num_channels, seq_length)
    sequences_tensor = torch.tensor(np.stack((time_sequences, length_sequences, protocol_sequences), axis=1), dtype=torch.float32)

    # Create sequences for labels
    labels = df['Label'].values
    labels_seq = create_sequences(labels, seq_length)
    labels_tensor = torch.tensor(labels_seq[:, -1], dtype=torch.long)  # Use the last element of each sequence as the label

    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(sequences_tensor, labels_tensor, test_size=0.2, shuffle=True, random_state=42)

    return X_train, X_test, Y_train, Y_test

# Try different sequence lengths
seq_lengths = [7, 15, 30, 40, 60]

for seq_length in seq_lengths:
    print(f"Processing data for seq_length = {seq_length}")
    X_train, X_test, Y_train, Y_test = process_data_for_seq_length(df, seq_length)


    # Initialize and train the classifier
    classifier = DecisionTreeClassifier(random_state=42)
    classifier.fit(X_train.view(X_train.shape[0], -1).numpy(), Y_train.numpy())

    # Predict and evaluate
    Y_pred = classifier.predict(X_test.view(X_test.shape[0], -1).numpy())
    accuracy = accuracy_score(Y_test.numpy(), Y_pred)
    print(f"Accuracy for seq_length = {seq_length}: {accuracy}")
    print(classification_report(Y_test.numpy(), Y_pred))

    print("\n")
    print("---------------------------------------------------------------")
    print("\n")

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
# Dividi i dati in caratteristiche (X) ed etichetta (Y)

#X = df.drop('Label')
X = df.drop(columns=['Label'])
X = X.drop('No.', axis=1)
X = X.drop('Source', axis=1)
X = X.drop('Destination', axis=1)
y = df['Label']

nan_mask = np.isnan(X)
#X_train_flat_np[nan_mask] = np.nanmean(X_train_flat_np)
X[nan_mask] = 0
# Addestra un modello di classificazione
rf_clf = RandomForestClassifier()
rf_clf.fit(X, y)

# Ottieni l'importanza delle caratteristiche
feature_importances = rf_clf.feature_importances_

# Crea un DataFrame con l'importanza delle caratteristiche
importances_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Ordina le caratteristiche per importanza
importances_df = importances_df.sort_values(by='Importance', ascending=False)

# Visualizza l'analisi di Pareto
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importances_df)
plt.title('Analisi di Pareto delle Caratteristiche con Random Forest')
plt.show()

from scipy.stats import kruskal

# Valuta l'importanza di ciascuna caratteristica utilizzando il test di Kruskal-Wallis
p_values = []

X = df.drop(columns=['Label'])
y = df['Label']

for column in X.columns:
    _, p_value = kruskal(*[X[column][y == label].dropna() for label in np.unique(y)])
    p_values.append(p_value)

# Crea un DataFrame con i p-value
kruskal_df = pd.DataFrame({
    'Feature': X.columns,
    'p-value': p_values
})

# Ordina le caratteristiche per p-value (valori più bassi indicano maggiore importanza)
kruskal_df = kruskal_df.sort_values(by='p-value')

# Visualizza i p-value
plt.figure(figsize=(10, 6))
sns.barplot(x='p-value', y='Feature', data=kruskal_df)
plt.title('Importanza delle Caratteristiche secondo Kruskal-Wallis')
plt.show()

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Separate features and target variable
X = df.drop('Label', axis=1)
X = X.drop('No.', axis=1)
X = X.drop('Source', axis=1)
X = X.drop('Destination', axis=1)
y = df['Label']

nan_mask = np.isnan(X)
#X_train_flat_np[nan_mask] = np.nanmean(X_train_flat_np)
X[nan_mask] = 0

# Apply Chi-Square test
chi2_selector = SelectKBest(chi2, k='all')
chi2_selector.fit(X, y)

# Get feature scores
chi2_scores = chi2_selector.scores_
feature_scores = pd.DataFrame({'Feature': X.columns, 'Chi2 Score': chi2_scores})
feature_scores = feature_scores.sort_values(by='Chi2 Score', ascending=False)
print("Chi-Square Feature Scores:")
print(feature_scores)

# Initialize the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model
rf_model.fit(X, y)
# Fit it for the bigger
rf.model.fit(X_sequences, Y_sequences)

# Get feature importances
importances = rf_model.feature_importances_
cum_importances = [0 for _ in range(len(importances))]
for i in range(len(importances)):
  index = i % len(importances)
  cum_importances += importances[i]

feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': cum_importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance using Random Forest')
plt.gca().invert_yaxis()  # To display the most important features at the top
plt.show()

print("Random Forest Feature Importances:")
print(feature_importances)
'''
