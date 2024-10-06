import streamlit as st
st.set_page_config(
    page_title="Bangalore House Price Prediction",
    page_icon="🏡",  # Unicode emoji or path to a custom icon
)

import numpy as np
import pickle
import json

# Add custom CSS to set the background image
st.markdown("""
    <style>
   #root > div:nth-child(1) > div.withScreencast > div > div > div > section > div.stAppViewBlockContainer.block-container.st-emotion-cache-13ln4jf.ea3mdgi5 > div > div > div > div.stForm.st-emotion-cache-qcpnpn.e10yg2by1{
            background-image:url("https://cdn4.vectorstock.com/i/1000x1000/35/93/wooden-background-template-natural-light-shade-vector-42133593.jpg");
            background-repeat: no-repeat;
            backgournd-op
            height: 100vh;}

            
    #bangalore-house-price-prediction{
            outline:5px;
            color : rgb(255,255,255);}

        body{
            overflow: hidden;
              }
    .stApp {
        background-image: url("https://potentpages.com/wp-content/uploads/2024/06/house.webp");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        height: 100vh; /* Full viewport height */
    }
          

    #root > div:nth-child(1) > div.withScreencast > div > div > div > section > div.stAppViewBlockContainer.block-container.st-emotion-cache-13ln4jf.ea3mdgi5 > div > div > div > div.stForm.st-emotion-cache-qcpnpn.e10yg2by1 > div > div > div > div:nth-child(2) > div > label > div > p{
        font-size : 25px;
        font-weight: bold;
        color : rgb(255,255,255 );
        background-color:rgb(39,39,49);
        padding:4px;
        border-radius:8px;
            }
    
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section > div.stAppViewBlockContainer.block-container.st-emotion-cache-13ln4jf.ea3mdgi5 > div > div > div > div.stForm.st-emotion-cache-qcpnpn.e10yg2by1 > div > div > div > div:nth-child(3) > div > label > div > p{
        font-size : 25px;
        font-weight: bold;
        color : rgb(255,255,255 );
        background-color:rgb(39,39,49);
        padding:4px;
        border-radius:8px;}

    #root > div:nth-child(1) > div.withScreencast > div > div > div > section > div.stAppViewBlockContainer.block-container.st-emotion-cache-13ln4jf.ea3mdgi5 > div > div > div > div.stForm.st-emotion-cache-qcpnpn.e10yg2by1 > div > div > div > div:nth-child(4) > div > label > div > p{
        font-size : 25px;
        font-weight: bold;
        color : rgb(255,255,255 );
        background-color:rgb(39,39,49);
        padding:4px;
        border-radius:8px;}

    #root > div:nth-child(1) > div.withScreencast > div > div > div > section > div.stAppViewBlockContainer.block-container.st-emotion-cache-13ln4jf.ea3mdgi5 > div > div > div > div.stForm.st-emotion-cache-qcpnpn.e10yg2by1 > div > div > div > div:nth-child(5) > div > label > div > p{
        font-size : 25px;
        font-weight: bold;
        color : rgb(255,255,255 );
        background-color:rgb(39,39,49);
        padding:4px;
        border-radius:8px;}


    #root > div:nth-child(1) > div.withScreencast > div > div > div > section > div.stAppViewBlockContainer.block-container.st-emotion-cache-13ln4jf.ea3mdgi5 > div > div > div > div:nth-child(3) > div > div > div > div > div > div > p{
        font-size : 25px;
        font-weight: bold;
        color : rgb(255,255,255 );
        background-color:rgb(39,39,49);
        padding:4px;
        border-radius:8px;}
            
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section > div.stAppViewBlockContainer.block-container.st-emotion-cache-13ln4jf.ea3mdgi5 > div > div > div > div:nth-child(3) > div > div{
            background-color: rgb(0,0,0);
             }
    </style>
            
    """, unsafe_allow_html=True,)


# Load the pre-trained model
with open('banglore_home_prices_model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_price(area, bhk, bath, location):
    # Assuming that the model takes input in the order [area, bhk, bath, location_encoded]
    loc_index = locations.index(location)
    x = np.zeros(len(locations))
    x[0] = area
    x[1] = bhk
    x[2] = bath
    x[loc_index] = 1
    return model.predict([x])[0]


# Create a Streamlit app interface
with st.form(key="Bangalore House Price Prediction"):

    st.title("Bangalore House Price Prediction")

# User input fields for the model
    area = st.number_input("Enter the area in square feet:", min_value=500, max_value=30000, step=50)
    bhk = st.number_input("Enter the number of BHK:", min_value=2, max_value=16, step=1)
    bath = st.number_input("Enter the number of bathrooms:", min_value=1, max_value=16, step=1)

# Location selection
    locations = [ "1st block jayanagar", "1st phase jp nagar", "2nd phase judicial layout", "2nd stage nagarbhavi", "5th block hbr layout", "5th phase jp nagar", "6th phase jp nagar", "7th phase jp nagar", "8th phase jp nagar", "9th phase jp nagar", "aecs layout", "abbigere", "akshaya nagar", "ambalipura", "ambedkar nagar", "amruthahalli", "anandapura", "ananth nagar", "anekal", "anjanapura", "ardendale", "arekere", "attibele", "beml layout", "btm 2nd stage", "btm layout", "babusapalaya", "badavala nagar", "balagere", "banashankari", "banashankari stage ii", "banashankari stage iii", "banashankari stage v", "banashankari stage vi", "banaswadi", "banjara layout", "bannerghatta", "bannerghatta road", "basavangudi", "basaveshwara nagar", "battarahalli", "begur", "begur road", "bellandur", "benson town", "bharathi nagar", "bhoganhalli", "billekahalli", "binny pete", "bisuvanahalli", "bommanahalli", "bommasandra", "bommasandra industrial area", "bommenahalli", "brookefield", "budigere", "cv raman nagar", "chamrajpet", "chandapura", "channasandra", "chikka tirupathi", "chikkabanavar", "chikkalasandra", "choodasandra", "cooke town", "cox town", "cunningham road", "dasanapura", "dasarahalli", "devanahalli", "devarachikkanahalli", "dodda nekkundi", "doddaballapur", "doddakallasandra", "doddathoguru", "domlur", "dommasandra", "epip zone", "electronic city", "electronic city phase ii", "electronics city phase 1", "frazer town", "gm palaya", "garudachar palya", "giri nagar", "gollarapalya hosahalli", "gottigere", "green glen layout", "gubbalala", "gunjur", "hal 2nd stage", "hbr layout", "hrbr layout", "hsr layout", "haralur road", "harlur", "hebbal", "hebbal kempapura", "hegde nagar", "hennur", "hennur road", "hoodi", "horamavu agara", "horamavu banaswadi", "hormavu", "hosa road", "hosakerehalli", "hoskote", "hosur road", "hulimavu", "isro layout", "itpl", "iblur village", "indira nagar", "jp nagar", "jakkur", "jalahalli", "jalahalli east", "jigani", "judicial layout", "kr puram", "kadubeesanahalli", "kadugodi", "kaggadasapura", "kaggalipura", "kaikondrahalli", "kalena agrahara", "kalyan nagar", "kambipura", "kammanahalli", "kammasandra", "kanakapura", "kanakpura road", "kannamangala", "karuna nagar", "kasavanhalli", "kasturi nagar", "kathriguppe", "kaval byrasandra", "kenchenahalli", "kengeri", "kengeri satellite town", "kereguddadahalli", "kodichikkanahalli", "kodigehaali", "kodigehalli", "kodihalli", "kogilu", "konanakunte", "koramangala", "kothannur", "kothanur", "kudlu", "kudlu gate", "kumaraswami layout", "kundalahalli", "lb shastri nagar", "laggere", "lakshminarayana pura", "lingadheeranahalli", "magadi road", "mahadevpura", "mahalakshmi layout", "mallasandra", "malleshpalya", "malleshwaram", "marathahalli", "margondanahalli", "marsur", "mico layout", "munnekollal", "murugeshpalya", "mysore road", "ngr layout", "nri layout", "nagarbhavi", "nagasandra", "nagavara", "nagavarapalya", "narayanapura", "neeladri nagar", "nehru nagar", "ombr layout", "old airport road", "old madras road", "padmanabhanagar", "pai layout", "panathur", "parappana agrahara", "pattandur agrahara", "poorna pragna layout", "prithvi layout", "r.t. nagar", "rachenahalli", "raja rajeshwari nagar", "rajaji nagar", "rajiv nagar", "ramagondanahalli", "ramamurthy nagar", "rayasandra", "sahakara nagar", "sanjay nagar", "sarakki nagar", "sarjapur", "sarjapur  road", "sarjapura - attibele road", "sector 2 hsr layout", "sector 7 hsr layout", "seegehalli", "shampura", "shivaji nagar", "singasandra", "somasundara palya", "sompura", "sonnenahalli", "subramanyapura", "sultan palaya", "tc palaya", "talaghattapura", "thanisandra", "thigalarapalya", "thubarahalli", "tindlu", "tumkur road", "ulsoor", "uttarahalli", "varthur", "varthur road", "vasanthapura", "vidyaranyapura", "vijayanagar", "vishveshwarya layout", "vishwapriya layout", "vittasandra", "whitefield", "yelachenahalli", "yelahanka", "yelahanka new town", "yelenahalli", "yeshwanthpur", "total_sqft", "bath", "bhk",]
    location = st.selectbox("Select the location:", locations)

    submit_button = st.form_submit_button("Predict Price")

# Predict button
if submit_button:
    prediction = predict_price(area, bhk, bath, location)
    st.success(f'The predicted price of the house is: ₹ {prediction:.2f} Lakhs')


# st.button("Scroll up to view the price.") 

file_path = 'banglore_home_prices_model.pkl'

try:
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error(f"The file {file_path} was not found.")
except pickle.UnpicklingError:
    st.error("Error occurred while unpickling the file.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
