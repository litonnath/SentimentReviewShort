# Define a function to take a series of reviews
# and predict whether each one is a positive or negative review

# max_length = 100 # previously defined
import tensorflow_datasets as tfds

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import load_model

import numpy as np

app=Flask(__name__)

from flask import Flask,request,jsonify,render_template

max_length = 50
def predict_review(model, new_sentences, maxlen=max_length, show_padded_sequence=True ):
    
  # Keep the original sentences so that we can keep using them later
  # Create an array to hold the encoded sequences
    

    vocab_size = 1000
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(new_sentences, vocab_size, max_subword_length=5)
    
    new_sequences = []

    # Convert the new reviews to sequences
    
    for i, frvw in enumerate(new_sentences):
        new_sequences.append(tokenizer.encode(frvw))

    trunc_type='post' 
    padding_type='post'

  # Pad all sequences for the new reviews

    new_reviews_padded = pad_sequences(new_sequences, maxlen=max_length, 
                                 padding=padding_type, truncating=trunc_type)             

    classes = model.predict(new_reviews_padded)

  # The closer the class is to 1, the more positive the review is

    for x in range(len(new_sentences)):
        if (show_padded_sequence):
            pass
#             print(new_reviews_padded[x])
            
    # Print the review as text
    
#     print(new_sentences[x])
    
    # Print its predicted class
    
#     print(str(new_sentences[x]))
#     print(classes[x][0])
    if (classes[x][0])>0.6:
        return ('Positive')
    else:
        return ('Negative')


    
@app.route('/')
def correct():
    return render_template('ShortSentenceReview.html')


@app.route('/Reviewinput', methods = ['POST'])  

def success():  
    #fname1,2,3 is a acutally a name 
        fake_reviews = request.form.get("fname1")
        
        
        history=load_model('NeuralNetwork.hdf5')
        
        Answer=predict_review(history, str(fake_reviews))
        
        
      
        return render_template("ShortSentenceReview.html", name = str(Answer))  
    
if __name__ == '__main__':  
    app.run()  



