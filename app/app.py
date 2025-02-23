# Import libraries
import pickle
import torch
from flask import Flask, render_template, request
from utils import *

# Choose CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#load pre-saved model weights
data = pickle.load(open('../models/bert_model_data.pkl','rb'))

vocab_size          = data['vocab_size']
word2id             = data['word2id']
batch_size          = data['batch_size']
max_mask            = data['max_mask']
max_len             = data['max_len']
n_layers            = data['n_layers']
n_heads             = data['n_heads']
d_model             = data['d_model']
d_ff                = data['d_ff']
d_k                 = data['d_k']
d_v                 = data['d_v']
n_segments          = data['n_segments']
word_list           = data['word_list']
id2word             = data['id2word']

# Initialize BERT model
model= BERT(
    n_layers, 
    n_heads, 
    d_model, 
    d_ff, 
    d_k, 
    n_segments, 
    vocab_size, 
    max_len, 
    device
)

# Model location
save_path = f'../models/S_BERT.pt'

#Load model
model.load_state_dict(torch.load(save_path, map_location=device))
model.eval()

# Set the model to the correct device
model.to(device)

# define classifier_head
classifier_head = torch.nn.Linear(vocab_size*3, 3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
optimizer_classifier = torch.optim.Adam(classifier_head.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])

def index():
    
    # Home page
    if request.method == 'GET':
        return render_template ('index.html', prompt = '')
    
    # Page after user input
    if request.method == 'POST':
        # get the user input
        
        sentence1 = request.form['sentence1']
        sentence2 = request.form['sentence2']
        
        result = predict_nli_and_similarity(model, classifier_head, sentence1, sentence2, device)
        similarity = result[0]  # Access the first element (similarity_score)
        nli_prediction = result[1]  # Access the second element (nli_result)


        return render_template('index.html', sentence1=sentence1, sentence2=sentence2,similarity=similarity,nli_prediction=nli_prediction)

if __name__ == '__main__':
    app.run(debug=True)