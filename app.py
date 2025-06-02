import flask
import string
from flask import Flask,request ,jsonify
import nltk
import pickle
import numpy as np
import sklearn
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

print("numpy:", np.__version__)
print("sklearn:", sklearn.__version__)
print("flask:",flask.__version__)
print("nltk:",nltk.__version__)
 

# model =pickle.load(open('nb_model.pkl','rb'))
# vectorizer =pickle.load(open('vectorizer.pkl','rb'))

model =pickle.load(open('model_final.pkl','rb'))
vectorizer =pickle.load(open('vectorizer_final.pkl','rb'))
# import pickle

# with open("phishing_model.pkl", "rb") as f:
#     url_model = pickle.load(f)

 
def transform_text (text):
     
    text = text.lower() 
    text = nltk.word_tokenize(text) 
    removedSC = list()
    for i in text:
        if i.isalnum():
            removedSC.append(i)
    text = removedSC[:]
    removedSWPC = list()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            removedSWPC.append(i)
    text = removedSWPC[:]
    ps = PorterStemmer()
    stemmed = list()
    for i in text:
        stemmed.append(ps.stem(i))
    
    text = stemmed[:]
    
    return " ".join(text)


# def extract_features(url):
#     """
#     Preprocess a single URL and extract relevant features for model prediction.
#     """
#     url = str(url).strip().lower()

#     # Extract features from URL
#     features = {
#         'url_length': len(url),
#         'num_digits': sum(c.isdigit() for c in url),
#         'num_special_chars': len(re.findall(r'[\W]', url)),
#         'contains_https': int('https' in url),
#         'contains_www': int('www' in url),
#         'num_subdomains': url.count('.'),
#     }
#     return features


# def preprocess_urls(urls):
#     """
#     Preprocess a list of URLs into model-ready features.
#     """
#     processed_data = []
#     for url in urls:
#         features = extract_features(url)
#         processed_data.append(features)
    
#     return pd.DataFrame(processed_data)








app =Flask(__name__)

@app.route('/')
def home():
  return "Hello World"

@app.route('/predict',methods=['POST'])
def predict():
   msg=request.form.get('msg')
   transformed_msg=transform_text(msg)
   vector_input=vectorizer.transform([transformed_msg])
   
   res=model.predict(vector_input)[0]
    
   result={'msg':msg,'result':str(res)}
   print(result)

   return jsonify(result)



# @app.route('/predict2', methods=['POST'])
# def predict():
#     # Get the URL from the request JSON
#     data = request.get_json()
#     url = data.get('url')

#     if url is None or url == '':
#         return jsonify({'error': 'No URL provided'}), 400

#     # Preprocess the URL
#     features = preprocess_urls([url])

#     # Predict using the model
#     prediction = url_model.predict(features)

#     # Decode the prediction (assuming '0' is 'Legitimate' and '1' is 'Phishing')
#     result = "Phishing" if prediction[0] == 1 else "Legitimate"

#     return jsonify({'prediction': result})


#Spam Detection APi run on port no. 8000
#to run this turn off the windows firewall defender public network then only run projrct.
         
if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8000, debug=True)
