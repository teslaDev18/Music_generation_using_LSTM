from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import collections
import pandas as pd
import os
import streamlit as st
import pretty_midi
import streamlit as st
from io import BytesIO

class MusicGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.1):
        super(MusicGenerator, self).__init__()        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        #self.dropout = nn.Dropout(p = 0.3)
        #self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        # Fully connected layers
        self.fcP = nn.Linear(hidden_size, output_size-2)
        self.reluP = nn.ReLU()
        self.fcS = nn.Linear(hidden_size, 1)
        self.reluS = nn.ReLU()
        self.fcD = nn.Linear(hidden_size,1)
        self.reluD = nn.ReLU()
        # Softmax layers for instrument and pitch predictions
        #self.softmax_instrument = nn.Softmax(dim=1)
        self.softmax_pitch = nn.Softmax(dim=1)
        
        self.init_weights()

    def init_weights(self):
        for layer in [self.fcP, self.fcS, self.fcD]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
        
    def forward(self, x):
        # LSTM layers
        out, _ = self.lstm1(x)
        #out= self.dropout(out)
        #out, _ = self.lstm2(out)
        #out, _ = self.lstm3(out)
        # Fully connected layers
        outP = self.fcP(out[:,-1,:])  # Take the last time step's output
        outP = self.reluP(outP)
        outS = self.fcS(out[:,-1,:])
        outS = self.reluS(outS)
        outD = self.fcD(out[:,-1,:])
        outD = self.reluD(outD)
        # Apply softmax to get probabilities for instrument prediction
        #instrument_probs = self.softmax_instrument(out[:, :128])
        # Apply softmax to get probabilities for pitch prediction
        pitch_probs = self.softmax_pitch(outP)

        # Keep step and duration as they are
        #step_duration = out[:, 256:]
        output = torch.cat((pitch_probs,outS,outD), dim=1)
        #output = pitch_probs
        #output = out
        
        return output

# Initialize your model
model = MusicGenerator(130,256,130)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.load_state_dict(torch.load('model.pth',map_location=device))
model.eval()

def midi_to_notes(midi_file):
    try:
        midi = pretty_midi.PrettyMIDI(midi_file)
    except:
        return pd.DataFrame([])
    
    notes = collections.defaultdict(list)

    # Get the list of instrument program numbers
    program_numbers = np.arange(0,128)
    
    # One-hot encode the instrument program numbers
    num_instruments = len(program_numbers)
    one_hot_encoded_instruments = np.eye(num_instruments)
    
    # Create a dictionary mapping each instrument's program number to its one-hot encoded vector
    program_to_one_hot = {program: one_hot_encoded_instruments[i] for i, program in enumerate(program_numbers)}
    
    all_notes = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            all_notes.append((note,instrument.program))
        
    sorted_notes = sorted(all_notes, key=lambda note: note[0].start)
    prev_start = sorted_notes[0][0].start

    for note_tuple in sorted_notes:
        note = note_tuple[0]
        program = note_tuple[1]
        start = note.start
        end = note.end
        # Append instrument data
        notes['instrument'].append(one_hot_encoded_instruments[program])
        notes['pitch'].append(one_hot_encoded_instruments[note.pitch])
        notes['start'].append(start)
        #print(start,end,program)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)

        prev_start = start

    return pd.DataFrame(notes)

def notes_to_midi(notes, output_file):

    midi_data = pretty_midi.PrettyMIDI()
    
    instruments_dict = {}

    prev_start = 0
    for note in notes:
        #instrument_number = note['instrument']
        #_, instrument_number = note[:128].max(dim=0)
        instrument_number = 0
        #print(instrument_number)
        instrument_number = int(instrument_number)
        
        if instrument_number not in instruments_dict:
            instruments_dict[instrument_number] = pretty_midi.Instrument(program=instrument_number)
        
        # Get the Instrument object from the dictionary
        instrument = instruments_dict[instrument_number]

        # Extract pitch, step, and duration from the note
        _, pitch = note[:128].max(dim=0)
        #pitch = note[-3]
        step = note[-2]
        #step = 0.4
        duration = note[-1]
        #duration = 0.3
        
        # Calculate start time and end time based on step and duration
        start_time = prev_start + step
        end_time = start_time + duration
        prev_start = start_time

        # Create a Note object and add it to the Instrument
        midi_note = pretty_midi.Note(
            velocity=100,  # Adjust the velocity (volume) of the note here if needed
            pitch=int(pitch),
            start=float(start_time),
            end=float(end_time)
        )
        #print(midi_note, start_time, end_time)
        instrument.notes.append(midi_note)

    # Add all instruments to the PrettyMIDI object
    for instrument_number, instrument in instruments_dict.items():
        midi_data.instruments.append(instrument)

    # Write the MIDI data to a file
    midi_data.write(output_file)

def pre_process(notes, sequence_length=50):
    # Scale pitch values
    scaled_notes = notes.copy()  # Create a copy to avoid modifying the original array
    # scaled_notes[:,-3] /= 128 
    # Initialize input and output arrays
    
    n = len(scaled_notes)
    note_input = np.zeros((max(0, n - sequence_length), sequence_length, scaled_notes.shape[1]))
    note_output = np.zeros((max(0, n - sequence_length), scaled_notes.shape[1]))
    #print(note_input.shape,note_output.shape,scaled_notes.shape,scaled_notes[:sequence_length].shape)
    # Populate input and output arrays
    for i in range(n - sequence_length):
        note_input[i] = scaled_notes[i:i + sequence_length]
        note_output[i] = scaled_notes[i + sequence_length]
    
    return note_input, note_output    

def fetch_files(files):
    all_notes = []
    for file in files:
        notes = midi_to_notes(file)
        all_notes.append(notes)
    all_notes = pd.concat(all_notes)

    #key_order = ['instrument','pitch', 'step', 'duration']
    key_order = ['pitch', 'step', 'duration']
    try:
        train_notes = np.stack([all_notes[key] for key in key_order], axis=1)
    except:
        return [],[]
    train_notes = train_notes.tolist()
    for i,train_note in enumerate(train_notes):
        train_notes[i] = np.concatenate((train_notes[i][0],train_notes[i][1:]))

    train_notes = np.array(train_notes)
    #print(type(train_notes))

    del all_notes
    #print(train_notes[15:20],"\nLength is: ",len(train_notes))

    note_input,note_output = pre_process(train_notes)
    del train_notes
    return note_input, note_output
    #print(note_input[0],"\nOUTPUT:\n",note_output[0])

def generate_notes(model, initial_notes, num_notes_to_generate):
      
    generated_notes = initial_notes.clone().to(device)
    model.eval()
    
    with torch.no_grad():  # Disable gradient calculation since we're only predicting
        for _ in range(num_notes_to_generate):
            #print(generated_notes.shape)
            input_sequence = generated_notes[-50:].unsqueeze(dim=0)
            
            # Predict the next note
            predictions = model(input_sequence)
            #_, predicted_instruments = predictions[:, :128].max(dim=1)
            #_, predicted_pitch = predictions[:, 128:256].max(dim=1)          
            _, predicted_pitch = predictions.max(dim=1)
            # Append the predicted note to the generated sequence
            generated_notes = torch.cat((generated_notes, predictions), dim=0)
    
    return generated_notes     

st.title("Music Generation")

uploaded_file = st.file_uploader("Choose a MIDI file", type=['mid', 'midi'])
file_name = [] 
if uploaded_file is not None:
    midi_data = BytesIO(uploaded_file.getvalue())
    file_name.append(midi_data)
    
num_notes_to_generate = st.number_input("Enter the number of notes to generate:", min_value=1, step=1, value=200)

if st.button("Generate Music"):
    if(len(file_name) == 0):
        st.error("Please upload a MIDI file.")
        st.stop()
    note_input_test, note_output_test = fetch_files(file_name)

    network_input_test_tensor = torch.tensor(note_input_test, dtype=torch.float32).to(device)
    network_output_test_tensor = torch.tensor(note_output_test, dtype=torch.float32).to(device)

    generated_sequence = generate_notes(model, network_input_test_tensor[0], num_notes_to_generate)
    notes_to_midi(generated_sequence, "Generated_Music.midi")
    midi_data = open("Generated_Music.midi", 'rb')

    file_path = "Generated_Music.midi"

    # Open the file in binary mode
    with open(file_path, "rb") as file:
        btn = st.download_button(
                label="Download MIDI file",
                data=file,
                file_name="Generated_Music.midi",
                mime="audio/midi")

    st.success("Music generated successfully!")
